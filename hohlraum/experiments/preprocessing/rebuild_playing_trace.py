"""Visible-content provider log parsing. Never invent prefix IDs from usage lengths."""
from pathlib import Path
from collections import Counter
import copy,json,sys
from transformers import AutoTokenizer
class Unsupported(Exception):pass
def text_content(value):
 if value is None:return ''
 if isinstance(value,str):return value
 if isinstance(value,list):
  chunks=[]
  for x in value:
   if isinstance(x,str):chunks.append(x);continue
   if x.get('type') in ['text','input_text','output_text','summary_text']:chunks.append(x.get('text',''))
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


# -*- coding: utf-8 -*-
"""Content-derived Qwen3 traces, with explicitly marked opaque token estimates.
No IDs or prefix boundaries are inferred from usage length changes.
"""
import copy,datetime,hashlib,json,re,sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
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


"""Paper-visible-text reconstruction using audited legacy parsers/template.

No per-epoch initial length calibration and no hidden-reasoning length inference.
Explicit reconstruction assumption: the mean first-request shortfall across
hours is fixed shared padding prepended before all visible messages. Residual
padding persists at its historical insertion position while that prefix survives. Replica namespaces are applied when exporting.
"""
import argparse
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import random
import re
import sys
from pathlib import Path




"""Persistent synthetic history; positions are offsets in unpadded token bytes.

Only exact retained visible prefixes justify retaining opaque historical padding.
A rewritten/compacted suffix has no evidence that its unknown content survived.
Fixed startup padding is supplied separately and is never part of this history.
"""
from array import array
import hashlib
import sys


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


class PaddedHistory:
    def __init__(self):
        self.visible = b''
        self.anchors = []

    def update(self, visible, shared, logged_tokens, identity):
        assert len(visible)%4 == len(shared)%4 == 0
        common = prefix_bytes(self.visible, visible)
        # An anchor at the retained prefix end belongs to that old history, and
        # must precede the newly appended text, not move to the new input end.
        kept = [(pos, data) for pos, data in self.anchors if pos <= common]
        retained = sum(len(data)//4 for _, data in kept)
        chunks = [shared]
        cursor = 0
        for pos, data in kept:
            chunks.extend((visible[cursor:pos], data))
            cursor = pos
        chunks.append(visible[cursor:])
        known_n = (len(shared)+len(visible))//4 + retained
        added = max(0, logged_tokens-known_n)
        if added:
            ids = array('I', [16+x for x in hashlib.sha256(identity.encode()).digest()])
            if sys.byteorder != 'little':
                ids.byteswap()
            data = (ids.tobytes()*((added+31)//32))[:added*4]
            chunks.append(data)
            if kept and kept[-1][0] == len(visible):
                kept[-1] = (len(visible), kept[-1][1]+data)
            else:
                kept.append((len(visible), data))
        self.anchors = kept
        self.visible = visible
        return b''.join(chunks), retained+added, added, retained


ORIGINAL_TEXT = text_content


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


def normalized_text(v):
    if isinstance(v, list):
        return '\n'.join(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                         if isinstance(x, dict) and x.get('type') == 'tool_reference'
                         else ORIGINAL_TEXT([x]) for x in v)
    return ORIGINAL_TEXT(v)


text_content = normalized_text


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


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
    return hashlib.sha256(json.dumps([sha(p) for p in files]).encode()).hexdigest()


def build_trajectory(args):
    raw, tokenizer_dir, output, identity = map(Path, args[:4])
    slice_seconds = args[4] if len(args) > 4 else None
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    identity = str(identity)
    dest = output / (identity.replace('/', '__') + '.jsonl')
    stats = Counter()
    hours = []
    excluded = []
    last_t = origin = None
    previous_hour_last = []
    index = 0

    def render(r):
        return TOKEN.sub('', tok.apply_chat_template(r['messages'], tokenize=False,
            tools=r.get('tools') or None, add_generation_prompt=True, enable_thinking=True))

    # Learn the shared length only from hourly first requests, never later calls.
    common_prefix = None
    first_requests = []
    for path in sorted((raw / identity).glob('hour_*/session.jsonl')):
        r = next(parser(path, identity.split('/')[0], Counter()))
        first = u32(tok.encode(render(r), add_special_tokens=False))
        if common_prefix is None:
            common_prefix = first
        else:
            limit = min(len(common_prefix), len(first))
            stop = next((j for j in range(0, limit, 4)
                         if common_prefix[j:j+4] != first[j:j+4]), limit)
            common_prefix = common_prefix[:stop]
        first_requests.append(dict(hour=int(path.parent.name.split('_')[1]),
            known_tokens=len(first)//4, logged_tokens=int(r['usage']),
            shortfall=max(0, int(r['usage'])-len(first)//4)))
    assert first_requests and common_prefix is not None
    # Round the nonnegative mean to nearest integer, with halves rounded up.
    total_gap = sum(r['shortfall'] for r in first_requests)
    shared_n = (2*total_gap+len(first_requests))//(2*len(first_requests))
    marker = hashlib.sha256(f'{identity}/shared-mean-padding-v5'.encode()).digest()
    pattern = u32([16+x for x in marker])
    shared = (pattern*((shared_n+31)//32))[:shared_n*4]
    shared_spec = dict(method='mean hourly first-request positive shortfall, rounded half up',
        assumption='Unknown fixed shared input, not recovered original text',
        common_visible_prefix_tokens=len(common_prefix)//4, shared_padding_tokens=shared_n,
        first_requests=first_requests)
    (output/(identity.replace('/', '__')+'.shared.json')).write_text(json.dumps(shared_spec,indent=2))

    @lru_cache(maxsize=8192)
    def encode_chunk(s):
        return u32(tok.encode(s, add_special_tokens=False))

    with dest.open('w') as out:
        for path in sorted((raw / identity).glob('hour_*/session.jsonl')):
            hour = int(path.parent.name.split('_')[1])
            # Retain the legacy explicit non-text exclusion instead of silently
            # pretending an image is empty text.
            has_image = any(re.search(r'"type"\s*:\s*"(?:input_image|image|image_url)"', l)
                            for l in path.open())
            if has_image:
                excluded.append(dict(hour=hour, reason='image_content', source_sha256=sha(path)))
                continue
            audit = Counter()
            hstats = Counter()
            previous = b''
            history = PaddedHistory()
            hashes = []
            first_hashes = None
            first_t = None
            for r in parser(path, identity.split('/')[0], audit):
                rendered = tok.apply_chat_template(r['messages'], tokenize=False,
                    tools=r.get('tools') or None,
                    add_generation_prompt=True, enable_thinking=True)
                # Legacy parser marks inaccessible reasoning. Its estimates are
                # intentionally NOT rendered as text or inserted as hidden KV.
                rendered = TOKEN.sub('', rendered)
                parts = re.split(r'(?=<\|im_start\|>)', rendered)
                visible = b''.join(encode_chunk(p) for p in parts)
                if hstats['requests'] < 2:
                    assert visible == u32(tok.encode(rendered, add_special_tokens=False))
                    stats['full_tokenizer_equivalence_checks'] += 1
                visible_n = len(visible) // 4
                # Fixed modeled input applies to EVERY call, including compaction.
                # Place fixed modeled input before variable conversation, so context
                # compaction cannot invalidate its prefix hashes.
                shared_used = shared_n
                current, padding, added_padding, retained_padding = history.update(
                    visible, shared, int(r['usage']),
                    f'{identity}/{hour}/{r["epoch"]}/{index}')
                n = len(current)//4
                assert len(current)//4 == n
                hi = min(len(previous), len(current))//4
                lo = 0
                if previous[:hi*4] == current[:hi*4]:
                    lo = hi
                else:
                    while lo < hi:
                        mid = (lo+hi+1)//2
                        if previous[:mid*4] == current[:mid*4]:
                            lo = mid
                        else:
                            hi = mid-1
                hashes = hashes[:lo//1024]
                prefix = bytes.fromhex(hashes[-1]) if hashes else b''
                while len(hashes) < n//1024:
                    j = len(hashes)*4096
                    prefix = hashlib.sha256(prefix+current[j:j+4096]).digest()
                    hashes.append(prefix.hex())
                t = r['t']
                if last_t is not None and t < last_t:
                    raise ValueError(f'Nonmonotone timestamps: {path}:{r["line"]}')
                origin = t if origin is None else origin
                if first_t is None:
                    first_t = t
                    first_hashes = list(hashes)
                row = dict(index=index, t=t-origin, observed_at=t, hour=hour,
                    source_line=r['line'], segment=r['epoch'], raw_input_tokens=n,
                    logged_input_tokens=r['usage'], visible_tokens=visible_n,
                    shared_padding_tokens=shared_used,
                    padding_tokens=padding, added_padding_tokens=added_padding,
                    retained_padding_tokens=retained_padding, think_time=0 if last_t is None else t-last_t,
                    prefix_hashes=list(hashes))
                out.write(json.dumps(row,separators=(',',':'))+'\n')
                index += 1
                hstats.update(requests=1, input_tokens=n, logged_tokens=r['usage'],
                    visible_tokens=visible_n, padding_tokens=padding,
                    added_padding_tokens=added_padding, retained_padding_tokens=retained_padding,
                    shared_padding_tokens=shared_used,
                    prefix_changed_requests=int(not visible.startswith(common_prefix)),
                    longer_than_logged=int(n>r['usage']))
                hstats['max_input_tokens'] = max(hstats['max_input_tokens'],n)
                previous = current
                last_t = t
                if slice_seconds is not None and t-origin > slice_seconds:
                    break
            common = 0
            for a,b in zip(previous_hour_last,first_hashes or []):
                if a != b:
                    break
                common += 1
            hours.append(dict(hour=hour, source_sha256=sha(path), stats=dict(hstats),
                parser_audit=dict(audit), first_observed_at=first_t, last_observed_at=last_t,
                common_prefix_with_previous_hour_last_tokens=common*1024))
            previous_hour_last = list(hashes)
            print(json.dumps(dict(trajectory=identity,hour=hour,requests=index)),flush=True)
            if slice_seconds is not None and last_t is not None and last_t-origin > slice_seconds:
                break
    report = dict(inputs_sha256=trajectory_inputs_sha256(raw,tokenizer_dir,identity),slice_seconds=slice_seconds,trajectory=identity,requests=index,hours=hours,excluded=excluded,checks=dict(stats),
        reconstruction_version='v7-persistent-history-padding', shared_spec=shared_spec,
        tools_sha256=sha(Path(__file__).parent/'tool_definitions.json')
        if identity.startswith('gpt') else None)
    dest.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    return report


def export(output, copies, seed, mean_arrival):
    files = sorted(output.glob('*__agent*.jsonl'))
    rng = random.Random(seed)
    schedule = [(p,copy) for p in files for copy in range(copies)]
    rng.shuffle(schedule)
    offer = 0
    count = 0
    roots = []
    with (output/'selected_rows.json').open('w') as f:
        f.write('[')
        for root,(path,copy) in enumerate(schedule):
            identity=path.stem.replace('__','/')
            sid=f'{identity}/replica_{copy:04d}'
            if root > 0:
                offer += rng.expovariate(1/mean_arrival)
            fold=int.from_bytes(hashlib.sha256(('fig8-crossfit:'+identity).encode()).digest()[:8],'big')%2
            mapping={}
            roots.append(dict(root=root,session_id=sid,source_trajectory=identity,root_offer=offer))
            for line in path.open():
                r=json.loads(line)
                bs=[]
                for h in r.pop('prefix_hashes'):
                    if h not in mapping:
                        mapping[h]=int.from_bytes(hashlib.sha256(sid.encode()+bytes.fromhex(h)).digest()[:8],'big')
                    bs.append(mapping[h])
                r.update(root=root,chain=0,owner=root,session_id=sid,source_trajectory=identity,
                         replica=copy,root_offer=offer,first_delay=0,blocks=bs,
                         reference_fold=fold,admission_fold=fold)
                if count:f.write(',\n')
                f.write(json.dumps(r,separators=(',',':')))
                count+=1
        f.write(']')
    (output/'sessions.json').write_text(json.dumps(roots,indent=2))
    return count,len(roots)


def balanced_export(trajectories, output, seed, mean_arrival=15):
    """Reproduce the published balanced pool and its deterministic 65-root selection.

    The historical namespace schedule is retained solely to preserve existing
    block IDs. It does not reweight the balanced candidate pool or select seeds.
    """
    files = sorted(trajectories.glob('*__agent*.jsonl'))
    by_identity = {p.stem.replace('__', '/'): p for p in files}
    rng = random.Random(20261013)
    schedule = []
    for provider, count, expected in [('claude_13agents', 161, 13),
                                      ('gpt_5agents', 46, 5), ('kimi_5agents', 23, 5)]:
        sources = sorted(k for k in by_identity if k.startswith(provider+'/'))
        if len(sources) != expected:
            raise ValueError('Incomplete original trajectories for '+provider)
        repeated = []
        while len(repeated) < count:
            turn = sources.copy(); rng.shuffle(turn); repeated.extend(turn)
        schedule.extend(repeated[:count])
    rng.shuffle(schedule)
    legacy_roots = {}
    for root, identity in enumerate(schedule):
        legacy_roots.setdefault(identity, root)
    slots = [0]
    for _ in range(194):
        slots.append(slots[-1] + rng.expovariate(1/mean_arrival))
    groups, pool = {}, []
    for identity, path in sorted(by_identity.items()):
        sid = f'{identity}/mix702010/replica_{legacy_roots[identity]:04d}'
        fold = int.from_bytes(hashlib.sha256(('fig8-crossfit:'+identity).encode()).digest()[:8], 'big') % 2
        mapping, rows = {}, []
        for line in path.open():
            row = json.loads(line)
            hashes = row.pop('prefix_hashes')
            blocks = []
            for h in hashes:
                if h not in mapping:
                    mapping[h] = int.from_bytes(hashlib.sha256(sid.encode()+bytes.fromhex(h)).digest()[:8], 'big')
                blocks.append(mapping[h])
            row.update(root=0, owner=0, chain=0, session_id=sid, source_trajectory=identity,
                       root_offer=0, first_delay=0, reference_fold=fold, admission_fold=fold,
                       blocks=blocks)
            row.pop('source_blocks', None)
            rows.append(row)
            if row['t'] > 14400:
                break  # Original experimental slice includes its first successor.
        groups[identity] = rows
        seen, total, hits = set(), 0, 0
        for row in rows:
            state = hashlib.sha256(); missing = False
            for block in row['blocks']:
                state.update(int(block).to_bytes(8, 'big')); key = state.digest()
                if key not in seen: missing = True
                if not missing: hits += 1024
                seen.add(key)
            total += row['raw_input_tokens']
        for copy in range(5 if identity.startswith('claude') else 13):
            pool.append(dict(root=len(pool), source_trajectory=identity,
                             session_id=f'{identity}/balanced195/replica_{copy:04d}',
                             requests=len(rows), theoretical_hit_rate=hits/total,
                             root_offer=slots[len(pool)]))
    assert len(pool) == 195
    order = random.Random(seed).sample(list(range(195)), 65)
    output.mkdir(parents=True, exist_ok=True)
    sessions, count = [], 0
    with (output/'calibration_rows.json').open('w') as stream:
        stream.write('[')
        for root, old in enumerate(order):
            t = slots[root]; offer = t*6 if t <= 600 else t+3000
            session = dict(pool[old], root=root, root_offer=offer, original_pool_root=old)
            sessions.append(session)
            for raw in groups[session['source_trajectory']]:
                row = dict(raw, root=root, owner=root, session_id=session['session_id'], root_offer=offer)
                if count: stream.write(',\n')
                stream.write(json.dumps(row, separators=(',', ':'))); count += 1
        stream.write(']')
    (output/'sessions.json').write_text(json.dumps(sessions, indent=2))
    manifest = dict(version='balanced195-source-blocks-removed', selection_seed=seed,
                    arrival_seed=20261013, roots=65, requests=count,
                    candidate_pool_counts=dict(claude=65, gpt=65, kimi=65),
                    copies_per_trajectory=dict(claude=5, gpt=13, kimi=13),
                    sampling='random.Random(selection_seed).sample(canonical195,65)',
                    candidate_order='lexicographic provider/trajectory, then ascending replica',
                    namespace='historical deterministic block IDs; runtime root isolates replicas',
                    slice='relative t <= 14400 plus first successor',
                    warmup_arrival_mean_s=mean_arrival*6, measurement_arrival_mean_s=mean_arrival,
                    provider_counts=dict(Counter(s['source_trajectory'].split('_')[0] for s in sessions)),
                    first30_qualified=sum(s['theoretical_hit_rate'] >= .985 for s in sessions[:30]),
                    development_seed_selection='272015-272017 were screened for at least 25 of first 30 roots with theoretical hit rate >= .985; arbitrary seeds are not screened',
                    calibration_rows_sha256=sha(output/'calibration_rows.json'))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--copies',type=int,default=10)
    p.add_argument('--layout',choices=['balanced','uniform'],default='balanced')
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--seed',type=int,default=272015)
    p.add_argument('--mean-arrival',type=float,default=15)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=a.resume)
    destination=a.output
    if a.layout=='balanced':
        a.output=destination/'_trajectories'
        a.output.mkdir(exist_ok=True)
    ids=[x.relative_to(a.raw) for prov in ('claude_13agents','gpt_5agents','kimi_5agents')
         for x in sorted((a.raw/prov).glob('agent*')) if x.is_dir()]
    def current_report(identity):
        path=a.output/(str(identity).replace('/','__')+'.report.json')
        if not path.exists():
            return False
        expected=sha(Path(__file__).parent/'tool_definitions.json') if str(identity).startswith('gpt') else None
        report=json.loads(path.read_text())
        return report.get('inputs_sha256') == trajectory_inputs_sha256(a.raw,a.tokenizer,str(identity)) and report.get('tools_sha256') == expected and report.get('reconstruction_version') == 'v7-persistent-history-padding' and report.get('slice_seconds') == (14400 if a.layout=='balanced' else None)
    jobs=[(a.raw,a.tokenizer,a.output,i,14400 if a.layout=='balanced' else None) for i in ids if not current_report(i)]
    with ProcessPoolExecutor(a.workers) as pool:
        for result in as_completed([pool.submit(build_trajectory,j) for j in jobs]):
            result.result()
    if a.layout=='balanced':
        for seed in a.seeds or [a.seed]:
            print(json.dumps(balanced_export(a.output,destination/f's{seed}',seed,a.mean_arrival)),flush=True)
        return
    count,roots=export(a.output,a.copies,a.seed,a.mean_arrival)
    manifest=dict(version='playing-text-v7-persistent-history-padding',seed=a.seed,copies=a.copies,trajectories=len(ids),
        roots=roots,requests=count,mean_root_arrival_s=a.mean_arrival,
        tokenizer_sha256=sha(a.tokenizer/'tokenizer.json'),template_sha256=sha(a.tokenizer/'tokenizer_config.json'),
        tools_sha256=sha(Path(__file__).parent/'tool_definitions.json'),
        tools_scope='GPT 0.130.0 write_stdin and update_plan only; not complete original tools',
        reconstruction='Qwen3 visible input plus fixed shared padding; historical padding retained at surviving visible-prefix anchors; append only positive length shortfall; discard anchors beyond rewritten/compacted prefix; never truncate retained input',
        shared_padding_assumption=True, shared_padding_position='before_all_visible_messages',
        namespace='SHA256(replica session_id, cumulative content prefix hash) truncated uint64',
        timing='Original observation-time deltas, retained legacy completion+delta replay; not measured pure think time',
        excluded='Non-text image hours explicitly listed in per-trajectory reports',
        no_prior_retraining=True, selected_rows_sha256=sha(a.output/'selected_rows.json'))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
