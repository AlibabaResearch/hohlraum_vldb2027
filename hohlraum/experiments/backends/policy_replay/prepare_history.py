"""Build a replay lifecycle header with the Figure 9(b) ablations."""
import sys
from pathlib import Path


def prepare_history(source, output):
    header = source.read_text()

    def replace_once(original, replacement):
        nonlocal header
        if header.count(original) != 1:
            raise ValueError(f"Serving lifecycle changed around {original!r}; review replay ablations.")
        header = header.replace(original, replacement, 1)

    # Gate the probability factors where TTD and admission both use them.
    # In full mode these hooks leave the production implementation unchanged.
    replace_once(
        "    double branch_score(unsigned fold,double idle)const{\n",
        "    double branch_score(unsigned fold,double idle)const{\n"
        "        if(replay_signal_==ReplaySignal::no_branch)return 1.;\n",
    )
    replace_once(
        "    double length_score(unsigned fold,unsigned qgs,unsigned length,double d)const {\n",
        "    double length_score(unsigned fold,unsigned qgs,unsigned length,double d)const {\n"
        "        if(replay_signal_==ReplaySignal::no_return)return 1.;\n",
    )
    replace_once(
        "private:\n    double threshold_, qg_gap_;",
        'private:\n'
        '    enum class ReplaySignal { full, no_return, no_branch };\n'
        '    static ReplaySignal replay_signal() {\n'
        '        const char* raw=std::getenv("HOHLRAUM_SIGNAL_ABLATION");\n'
        '        const std::string value=raw?raw:"full";\n'
        '        if(value=="full")return ReplaySignal::full;\n'
        '        if(value=="no_return")return ReplaySignal::no_return;\n'
        '        if(value=="no_branch")return ReplaySignal::no_branch;\n'
        '        throw std::invalid_argument("unknown replay lifecycle ablation: "+value);\n'
        '    }\n'
        '    const ReplaySignal replay_signal_=replay_signal();\n'
        '    double threshold_, qg_gap_;',
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header)


if __name__ == "__main__":
    prepare_history(Path(sys.argv[1]), Path(sys.argv[2]))
