"""Reconstruct page-aligned Playing histories from visible provider logs.

Each hourly session starts a new cache namespace. Provider cache reads bound
shared pages retained from an identical Qwen prefix between branching calls.
Within a segment, a verified append retains all preceding complete pages.
"""
import copy
import datetime
import hashlib
import json
import re
import sys
from array import array
from collections import Counter
from functools import lru_cache
from pathlib import Path

from hohlraum.experiments.resources import digest

PAGE = 1024
CONTRACT = "hourly-page-aligned-qwen-boundary-v2"

class Unsupported(Exception):pass
def text_content(value):
 if value is None:return ''
 if isinstance(value,str):return value
 if isinstance(value,list):
  chunks=[]
  for x in value:
   if isinstance(x,str):chunks.append(x);continue
   if x.get('type') in ['text','input_text','output_text','summary_text']:chunks.append(x.get('text',''))
   elif x.get('type')=='tool_reference':chunks.append(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')))
   else:raise Unsupported('non-text content: '+str(x.get('type')))
  return '\n'.join(chunks)
 return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def messages(m):
 role=m.get('role','user');role='system' if role=='developer' else role
 parts=m.get('content','');out=[];body=[];thinking=[];calls=[]
 if isinstance(parts,list):
  for x in parts:
   t=x.get('type')
   if t in ['text','input_text','output_text','summary_text']:body.append(x.get('text',''))
   elif t in ['thinking','think']:thinking.append(x.get('thinking',x.get('think','')))
   elif t=='tool_use':calls.append(dict(id=x['id'],type='function',function=dict(name=x['name'],arguments=x.get('input',{}))))
   elif t=='tool_result':out.append(dict(role='tool',content=text_content(x.get('content','')),tool_call_id=x.get('tool_use_id')))
   elif t=='redacted_thinking':raise Unsupported('redacted reasoning')
   else:raise Unsupported('non-text content: '+str(t))
 else:body=[text_content(parts)]
 for c in m.get('tool_calls',m.get('toolCalls',[])) or []:
  if 'function' in c:calls.append(c)
  else:calls.append(dict(id=c.get('id'),type='function',function=dict(name=c['name'],arguments=c.get('arguments',c.get('args',{})))))
 if body or thinking or calls or not out:
  x=dict(role=role,content='\n'.join(body))
  if thinking:x['reasoning_content']='\n'.join(thinking)
  if calls:x['tool_calls']=calls
  out.insert(0,x)
 return out
def claude(path,audit):
 nodes={};seen=set();epoch=0
 for line,s in enumerate(path.open(),1):
  x=json.loads(s)
  if x.get('subtype')=='compact_boundary':epoch+=1
  uid=x.get('uuid')
  if uid:nodes[uid]=x
  if x.get('type')!='assistant' or x.get('isSidechain'):continue
  m=x.get('message',{});mid=m.get('id');usage=m.get('usage',{});length=sum(usage.get(k,0) or 0 for k in ['input_tokens','cache_creation_input_tokens','cache_read_input_tokens'])
  if not mid or mid in seen or length<=0 or m.get('model')=='<synthetic>':continue
  seen.add(mid);ancestry=[];at=x.get('parentUuid');visited=set();missing=False
  while at:
   if at in visited:raise Unsupported('parent cycle')
   visited.add(at);n=nodes.get(at)
   if n is None:missing=True;break
   if n.get('subtype')=='compact_boundary':break
   msg=n.get('message')
   if msg and n.get('type') in ['user','assistant'] and msg.get('id')!=mid:ancestry.append(msg)
   at=n.get('parentUuid')
  if missing:audit['missing_parent_snapshots']+=1;continue
  merged=[]
  for msg in reversed(ancestry):
   if merged and msg.get('id') and msg.get('id')==merged[-1].get('id'):
    if not isinstance(msg.get('content'),list) or not isinstance(merged[-1].get('content'),list):raise Unsupported('non-list assistant chunks')
    merged[-1]['content'].extend(copy.deepcopy(msg['content']))
   else:merged.append(copy.deepcopy(msg))
  normalized=[]
  for msg in merged:normalized.extend(messages(msg))
  if normalized:yield line,x.get('timestamp'),epoch,normalized,length
def codex_item(p,audit):
 t=p.get('type')
 if t=='message':return messages(p)
 if t in ['function_call','custom_tool_call']:
  return [dict(role='assistant',content='',tool_calls=[dict(id=p.get('call_id'),type='function',function=dict(name=p['name'],arguments=p.get('arguments',p.get('input',''))))])]
 if t in ['function_call_output','custom_tool_call_output']:return [dict(role='tool',content=text_content(p.get('output','')),tool_call_id=p.get('call_id'))]
 if t=='reasoning':
  if p.get('encrypted_content'):audit['unavailable_encrypted_reasoning_items']+=1
  visible=p.get('content') or p.get('summary') or []
  return [dict(role='assistant',content='',reasoning_content=text_content(visible))] if visible else []
 raise Unsupported('unknown response item '+str(t))


TOKEN=re.compile(r'__HOHLRAUM_OPAQUE_[0-9a-f]{64}__')
def numbered_lines(path):
 with path.open() as f:yield from enumerate(f,1)
def stamp(x):return x/1000 if isinstance(x,(int,float)) else datetime.datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
def marker(key,n,opaque):
 h=hashlib.sha256(key.encode()).hexdigest();s='__HOHLRAUM_OPAQUE_'+h+'__';opaque[s]=n;return s
def norm_item(p,audit,opaque,key):
 if p['type']=='compaction':
  audit['encrypted_compaction_items']+=1
  return [dict(role='user',content=marker(key,1,opaque))]
 if p['type']=='reasoning':
  visible=text_content(p.get('content')or p.get('summary')or[])
  return [dict(role='assistant',content='',reasoning_content=visible)] if visible else []
 return codex_item(p,audit)
def gpt(path,audit,opaque):
 history=[];pending=None;totals=set();epoch=0;encrypted=[];response_start=0
 for line,s in numbered_lines(path):
  x=json.loads(s);t=x.get('type');p=x.get('payload')or{}
  if t=='session_meta':
   v=p.get('base_instructions');v=v.get('text','') if isinstance(v,dict) else v
   if v:history.append(dict(role='system',content=v))
  elif t=='compacted':
   replacement=p.get('replacement_history')
   if not isinstance(replacement,list):raise Unsupported('missing GPT replacement_history')
   # Base instructions are retained as system context; replacement_history is the model history.
   system=[m for m in history[:1] if m['role']=='system'];history=system
   for j,item in enumerate(replacement):history.extend(norm_item(item,audit,opaque,f'{path}:{line}:{j}'))
   epoch+=1;pending=None;encrypted=[];audit['explicit_compactions']+=1
  elif t=='response_item':
   isout=p.get('type') in ['reasoning','function_call','custom_tool_call'] or p.get('role')=='assistant'
   if isout and pending is None:pending=list(history);response_start=len(history);encrypted=[]
   if p.get('type')=='reasoning' and p.get('encrypted_content'):
    # Count will be supplied by this response's explicit reasoning_output_tokens.
    m=dict(role='assistant',content='',reasoning_content='');history.append(m);encrypted.append((m,f'{path}:{line}'));audit['encrypted_reasoning_items']+=1
   else:history.extend(norm_item(p,audit,opaque,f'{path}:{line}'))
  elif t=='event_msg' and p.get('type')=='token_count':
   info=p.get('info')or{};usage=info.get('last_token_usage')or{};key=json.dumps(info.get('total_token_usage',{}),sort_keys=True)
   if key in totals:continue
   totals.add(key)
   if pending is None or not usage.get('input_tokens'):audit['usage_without_pending_input']+=1;continue
   snapshot=pending;pending=None
   yield dict(line=line,t=stamp(x['timestamp']),epoch=epoch,messages=snapshot,usage=usage['input_tokens'])
   n=usage.get('reasoning_output_tokens',0)or 0
   if encrypted:
    # Aggregate reasoning at its first logged item; exact encrypted internal placement is unavailable.
    encrypted[0][0]['reasoning_content']=marker(encrypted[0][1],n,opaque);audit['opaque_reasoning_tokens']+=n
   elif n:
    history.insert(response_start,dict(role='assistant',content='',reasoning_content=marker(f'{path}:response{line}',n,opaque)));audit['reasoning_without_logged_item']+=1;audit['opaque_reasoning_tokens']+=n
   encrypted=[]
def kimi(path,audit,opaque):
 history=[];pending=None;assistant=None;tools=[];epoch=0;last_step=None
 for line,s in numbered_lines(path):
  x=json.loads(s);t=x.get('type');e=x.get('event')or{}
  if t=='context.apply_compaction':
   if 'keptUserMessageCount' in x:raise Unsupported('new Kimi compaction schema')
   count=x['compactedCount']
   if count>len(history):raise Unsupported(f'Kimi compaction count {count} exceeds history {len(history)}')
   history=[dict(role='user',content=x.get('contextSummary',x['summary']))]+history[count:]
   epoch+=1;pending=None;assistant=None;tools=[];audit['explicit_compactions']+=1;audit['compaction_retained_messages']+=len(history)-1
  elif t=='context.append_message':history.extend(messages(x['message']))
  elif t=='context.append_loop_event':
   typ=e.get('type')
   if typ=='step.begin':
    pending=list(history);assistant=dict(role='assistant',content='',reasoning_content='',tool_calls=[]);tools=[];last_step=e.get('uuid');history.append(assistant)
   elif typ=='content.part' and assistant is not None:
    part=e.get('part')or{};kind=part.get('type')
    if kind=='think':assistant['reasoning_content']+=part.get('think','')
    elif kind=='text':assistant['content']+=part.get('text','')
    else:raise Unsupported('unknown Kimi content '+str(kind))
   elif typ=='tool.call' and assistant is not None:assistant['tool_calls'].append(dict(id=e.get('toolCallId'),type='function',function=dict(name=e['name'],arguments=e.get('args',{}))))
   elif typ=='tool.result':history.append(dict(role='tool',content=text_content((e.get('result')or{}).get('output','')),tool_call_id=e.get('toolCallId')))
   elif typ=='step.end' and assistant is not None:
    # Kimi materializes the assistant at step.begin, even for an interrupted empty step.
    assistant=None;tools=[]
  elif t=='usage.record' and pending is not None:
   usage=x.get('usage')or{};length=sum(usage.get(k,0)or 0 for k in ['inputOther','inputCacheRead','inputCacheCreation'])
   if length:yield dict(line=line,t=stamp(x['time']),epoch=epoch,messages=pending,usage=length)
   pending=None


def prefix_bytes(a, b):
    lo, hi = 0, min(len(a), len(b)) // 4
    if a[:hi*4] == b[:hi*4]:
        return hi*4
    while lo < hi:
        mid = (lo+hi+1)//2
        if a[:mid*4] == b[:mid*4]:
            lo = mid
        else:
            hi = mid-1
    return lo*4


# Positions are offsets in visible-token bytes. Only an exact retained
# visible prefix carries its estimated padding forward within the same hour.
class PaddedHistory:
    def __init__(self):
        self.visible = b''
        self.anchors = []

    def update(self, visible, logged_tokens, identity):
        assert len(visible) % 4 == 0
        common = prefix_bytes(self.visible, visible)
        # An anchor at the retained prefix end belongs to that old history, and
        # must precede the newly appended text, not move to the new input end.
        kept = [(pos, data) for pos, data in self.anchors if pos <= common]
        retained = sum(len(data)//4 for _, data in kept)
        chunks = []
        cursor = 0
        for pos, data in kept:
            chunks.extend((visible[cursor:pos], data))
            cursor = pos
        chunks.append(visible[cursor:])
        known_n = len(visible)//4 + retained
        added = max(0, logged_tokens-known_n)
        if added:
            pattern = u32(16 + x for x in hashlib.sha256(identity.encode()).digest())
            data = (pattern * ((added + 31) // 32))[:added * 4]
            chunks.append(data)
            if kept and kept[-1][0] == len(visible):
                kept[-1] = (len(visible), kept[-1][1]+data)
            else:
                kept.append((len(visible), data))
        self.anchors = kept
        self.visible = visible
        return b''.join(chunks), retained+added, added, retained


def verified_tools(path, provider):
    """A partial, version-pinned recovery; never substitute current tool schemas."""
    if not provider.startswith('gpt'):
        return []
    version = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('type') == 'session_meta':
                version = row['payload'].get('cli_version')
                break
    bundle = json.loads((Path(__file__).parent/'tool_definitions.json').read_text())
    if version != bundle['client_version']:
        raise ValueError(f'Unverified tool version {version}: {path}')
    return bundle['tools']



def u32(ids):
    a = array('I', ids)
    if sys.byteorder != 'little':
        a.byteswap()
    return a.tobytes()


def parser(path, provider, audit):
    tools = verified_tools(path, provider)
    if provider.startswith('claude'):
        for line, t, epoch, msgs, usage in claude(path, audit):
            yield dict(line=line, t=stamp(t), epoch=epoch, messages=msgs, usage=usage)
    else:
        fn = gpt if provider.startswith('gpt') else kimi
        system = None
        if provider.startswith('kimi'):
            for line in path.open():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('type') == 'config.update' and 'systemPrompt' in row:
                    system = row['systemPrompt']
                    break
        for r in fn(path, audit, {}):
            r = dict(r, tools=tools)
            if system:
                r = dict(r, messages=[dict(role='system', content=system)] + r['messages'])
            yield r


def trajectory_inputs_sha256(raw, tokenizer_dir, identity):
    files = sorted((Path(raw)/identity).glob('hour_*/session.jsonl'))
    files += [Path(tokenizer_dir)/name for name in
              ['tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json']]
    files.append(Path(__file__).parent/'tool_definitions.json')
    return hashlib.sha256(json.dumps([digest(p) for p in files]).encode()).hexdigest()


def provider_cache_reads(path, provider):
    """Explicit provider cache-read counts keyed by raw-log line number."""
    result = {}
    for line, text in numbered_lines(path):
        row = json.loads(text)
        usage = None
        field = None
        if provider.startswith('claude') and row.get('type') == 'assistant':
            usage = (row.get('message') or {}).get('usage') or {}
            field = 'cache_read_input_tokens'
        elif provider.startswith('gpt') and row.get('type') == 'event_msg':
            payload = row.get('payload') or {}
            if payload.get('type') == 'token_count':
                usage = (payload.get('info') or {}).get('last_token_usage') or {}
                field = 'cached_input_tokens'
        elif provider.startswith('kimi') and row.get('type') == 'usage.record':
            usage = row.get('usage') or {}
            field = 'inputCacheRead'
        if usage is not None and field in usage:
            value = usage[field]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f'Invalid provider cache count: {path}:{line}')
            result[line] = value
    return result


def cache_read_boundary(provider, cached, logged, unpadded):
    """Translate a cache-read count into the token unit used for page alignment."""
    if not (0 <= cached <= logged and logged > 0 and unpadded > 0):
        raise ValueError('Invalid provider cache-read count or input length')
    if provider.startswith('claude'):
        return cached
    if provider.startswith(('gpt', 'kimi')):
        return cached * unpadded // logged
    raise ValueError(f'Unsupported Playing provider: {provider}')


def shared_pages(previous, current):
    for index, (before, after) in enumerate(zip(previous, current)):
        if before != after:
            return index
    return min(len(previous), len(current))


def page_align_hourly_requests(rows, cache_reads, provider, *, audit=None):
    """Export serving pages and the provider-bounded history used for priors."""
    audit = audit if audit is not None else Counter()
    previous_hour, previous_segment, previous_length = None, None, 0
    previous_content, previous_blocks, previous_prior = [], [], []
    next_block = 1
    for source in rows:
        row = dict(source)
        hour, line = int(row['hour']), int(row['source_line'])
        if (hour, line) not in cache_reads:
            raise ValueError(f'Missing explicit provider cache count: hour {hour}, line {line}')
        cached = cache_reads[hour, line]
        content = row.pop('prefix_hashes')
        original = int(row['raw_input_tokens'])
        if len(content) != original // PAGE:
            raise ValueError('Visible-history hash count does not match complete input pages')
        common = shared_pages(previous_content, content) if hour == previous_hour else 0
        boundary = cache_read_boundary(provider, cached, row['logged_input_tokens'], original)
        prior_keep = min(common, boundary // PAGE)
        segment = row['segment']
        append = (hour == previous_hour and segment == previous_segment
                  and original >= previous_length and common == len(previous_content))
        keep = common if append else prior_keep
        pages = (original + PAGE - 1) // PAGE
        prior_blocks = previous_prior[:prior_keep] + list(
            range(next_block, next_block + pages - prior_keep))
        # Restoring a complete append changes only identities on the reused
        # prefix. Fresh suffix identities stay those of the prior history.
        blocks = previous_blocks[:keep] + prior_blocks[keep:]
        next_block += pages - prior_keep
        row.update(blocks=blocks, pre_alignment_input_tokens=original,
                   raw_input_tokens=pages * PAGE, provider_cache_read_tokens=cached,
                   expected_previous_hit_tokens=keep * PAGE,
                   prior_blocks=prior_blocks, complete_append=append)
        if provider.startswith(('gpt', 'kimi')):
            row['provider_cache_read_qwen_estimate_tokens'] = boundary
        audit['requests'] += 1
        audit['allocated_tokens'] += pages * PAGE
        audit['new_tokens'] += (pages - keep) * PAGE
        audit['retained_tokens'] += keep * PAGE
        audit['append_retained_tokens'] += (keep - prior_keep) * PAGE
        audit['partial_tail_requests'] += int(original % PAGE != 0)
        audit['usage_cut_requests'] += int(prior_keep < common)
        audit['hours'] += int(hour != previous_hour)
        previous_segment, previous_length, previous_prior = segment, original, prior_blocks
        previous_hour, previous_content, previous_blocks = hour, content, blocks
        yield row


def build_trajectory(raw, tokenizer_dir, output, identity, *, hours):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    provider = identity.split('/')[0]
    paths = sorted((raw / identity).glob('hour_*/session.jsonl'))[:hours]
    if len(paths) != hours:
        raise ValueError(f'{identity} has fewer than {hours} recorded hours')
    dest = output / (identity.replace('/', '__') + '.jsonl')
    hour_reports = []
    last_t = origin = None
    index = 0

    @lru_cache(maxsize=8192)
    def encode_chunk(text):
        return u32(tok.encode(text, add_special_tokens=False))

    with dest.open('w') as out:
        for path in paths:
            hour = int(path.parent.name.split('_')[1])
            if any(re.search(r'"type"\s*:\s*"(?:input_image|image|image_url)"', line)
                   for line in path.open()):
                raise Unsupported(f'Image content in {path}')
            audit = Counter()
            counts = Counter()
            history = PaddedHistory()
            hashes = []
            previous = b''
            first_t = None
            for r in parser(path, provider, audit):
                rendered = tok.apply_chat_template(r['messages'], tokenize=False,
                    tools=r.get('tools') or None,
                    add_generation_prompt=True, enable_thinking=True)
                # Placeholder reasoning markers are not rendered as recovered text.
                rendered = TOKEN.sub('', rendered)
                parts = re.split(r'(?=<\|im_start\|>)', rendered)
                visible = b''.join(encode_chunk(part) for part in parts)
                if counts['requests'] < 2:
                    assert visible == u32(tok.encode(rendered, add_special_tokens=False))
                current, padding, added, retained = history.update(
                    visible, int(r['usage']), f'{identity}/{hour}/{r["epoch"]}/{index}')
                n = len(current) // 4
                hashes = hashes[:prefix_bytes(previous, current) // (4 * PAGE)]
                prefix = bytes.fromhex(hashes[-1]) if hashes else b''
                while len(hashes) < n // PAGE:
                    start = len(hashes) * 4 * PAGE
                    prefix = hashlib.sha256(prefix + current[start:start + 4 * PAGE]).digest()
                    hashes.append(prefix.hex())
                t = r['t']
                if last_t is not None and t < last_t:
                    raise ValueError(f'Nonmonotone timestamps: {path}:{r["line"]}')
                origin = t if origin is None else origin
                first_t = t if first_t is None else first_t
                row = dict(index=index, t=t-origin, observed_at=t, hour=hour,
                    source_line=r['line'], segment=r['epoch'], raw_input_tokens=n,
                    logged_input_tokens=r['usage'], visible_tokens=len(visible) // 4,
                    shared_padding_tokens=0, padding_tokens=padding,
                    added_padding_tokens=added, retained_padding_tokens=retained,
                    think_time=0 if last_t is None else t-last_t, prefix_hashes=list(hashes))
                out.write(json.dumps(row, separators=(',', ':')) + '\n')
                index += 1
                counts.update(requests=1, input_tokens=n, logged_tokens=r['usage'],
                              visible_tokens=len(visible) // 4, padding_tokens=padding)
                previous, last_t = current, t
            if first_t is None:
                raise ValueError(f'No complete logged request in {path}')
            hour_reports.append(dict(hour=hour, source_sha256=digest(path), counts=dict(counts),
                                     parser_audit=dict(audit), first_observed_at=first_t,
                                     last_observed_at=last_t))
    return dict(trajectory=identity, requests=index, hours=hour_reports,
                inputs_sha256=trajectory_inputs_sha256(raw, tokenizer_dir, identity),
                contract=CONTRACT)
