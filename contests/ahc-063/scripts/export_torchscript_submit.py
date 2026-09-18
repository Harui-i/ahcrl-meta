"""Export a slot-fusion AHC063 checkpoint and a matching C++ submit."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ahcrl.contests.ahc063.encoder import (
    ACTION_COUNT,
    ACTION_FEATURE_COUNT,
    BOARD_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    MAX_BOARD_SIZE,
    MAX_SEQUENCE_LENGTH,
)
from ahcrl.contests.ahc063.model import PPOModel

ROOT = Path(__file__).resolve().parents[3]
MAX_COLORS = 7
BASE91_ALPHABET = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~"'
)


class ExportPolicy(nn.Module):
    def __init__(self, model: nn.Module, apply_softmax: bool = False) -> None:
        super().__init__()
        self.model = model
        self.apply_softmax = apply_softmax

    def forward(
        self,
        board_food: torch.Tensor,
        board_features: torch.Tensor,
        slot_colors: torch.Tensor,
        slot_positions: torch.Tensor,
        global_features: torch.Tensor,
        previous_action: torch.Tensor,
        action_colors: torch.Tensor,
        action_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(
            board_food,
            board_features,
            slot_colors,
            slot_positions,
            global_features,
            previous_action,
            action_colors,
            action_features,
            mask,
        )
        return torch.softmax(logits, dim=-1) if self.apply_softmax else logits


def base91_encode(data: bytes) -> str:
    out: list[str] = []
    bit_queue = 0
    bit_count = 0
    for byte in data:
        bit_queue |= byte << bit_count
        bit_count += 8
        if bit_count > 13:
            value = bit_queue & 8191
            if value > 88:
                bit_queue >>= 13
                bit_count -= 13
            else:
                value = bit_queue & 16383
                bit_queue >>= 14
                bit_count -= 14
            out.extend((BASE91_ALPHABET[value % 91], BASE91_ALPHABET[value // 91]))
    if bit_count:
        out.append(BASE91_ALPHABET[bit_queue % 91])
        if bit_count > 7 or bit_queue > 90:
            out.append(BASE91_ALPHABET[bit_queue // 91])
    return "".join(out)


def c_string_chunks(value: str, width: int = 120) -> str:
    chunks = []
    for start in range(0, len(value), width):
        chunk = value[start : start + width]
        chunk = chunk.replace("\\", "\\\\").replace('"', '\\"').replace("?", "\\?")
        chunks.append(f'    "{chunk}"')
    return "\n".join(chunks)


def _config_value(config: dict[str, Any], name: str, default: int | str) -> int | str:
    return config.get(f"model_{name}", config.get(name, default))


def load_policy(
    checkpoint: Path, config: dict[str, Any], apply_softmax: bool = False
) -> ExportPolicy:
    if "model_channels" in config or "model_blocks" in config:
        raise ValueError("旧AHC063 checkpoint/configはslot_fusion_conv_v1へ移行できません")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PPOModel(
        architecture=str(_config_value(config, "architecture", "slot_fusion_conv_v1")),
        sequence_channels=int(_config_value(config, "sequence_channels", 64)),
        sequence_blocks=int(_config_value(config, "sequence_blocks", 8)),
        spatial_channels=int(_config_value(config, "spatial_channels", 128)),
        spatial_blocks=int(_config_value(config, "spatial_blocks", 4)),
        head_channels=int(_config_value(config, "head_channels", 128)),
        head_blocks=int(_config_value(config, "head_blocks", 2)),
    ).float()
    model_state = state.get("model")
    if not isinstance(model_state, dict) or not any(
        str(name).startswith("policy.encoder.token_encoder") for name in model_state
    ):
        raise ValueError("checkpoint is not a slot_fusion_conv_v1 AHC063 checkpoint")
    model.load_state_dict(model_state)
    return ExportPolicy(model.policy.eval(), apply_softmax).eval()


def export_torchscript(
    checkpoint: Path, config: dict[str, Any], apply_softmax: bool = False
) -> bytes:
    policy = load_policy(checkpoint, config, apply_softmax)
    dummy = (
        torch.zeros((1, MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=torch.uint8),
        torch.zeros((1, BOARD_FEATURE_COUNT, MAX_BOARD_SIZE, MAX_BOARD_SIZE)),
        torch.zeros((1, MAX_SEQUENCE_LENGTH, 2), dtype=torch.uint8),
        torch.full((1, MAX_SEQUENCE_LENGTH, 2), 255, dtype=torch.uint8),
        torch.zeros((1, GLOBAL_FEATURE_COUNT)),
        torch.zeros((1, 1), dtype=torch.uint8),
        torch.zeros((1, ACTION_COUNT), dtype=torch.uint8),
        torch.zeros((1, ACTION_COUNT, ACTION_FEATURE_COUNT)),
        torch.ones((1, ACTION_COUNT), dtype=torch.uint8),
    )
    with torch.no_grad():
        traced = torch.jit.trace(policy, dummy, strict=False)
        frozen = torch.jit.freeze(traced)
    output = Path("/tmp/ahc063_export.pt")
    frozen.save(str(output))
    return output.read_bytes()


CPP_TEMPLATE = r"""#include <ATen/Parallel.h>
#include <torch/script.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
namespace {
constexpr int MAX_N=16, MAX_COLORS=7, MAX_M=192, ACTIONS=4, FEATURES=8, ACTION_FEATURES=7;
constexpr int MAX_STEPS_PER_CELL=@MAX_STEPS_PER_CELL@;
const string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~\"";
const string model_data=
@MODEL@
;
vector<unsigned char> decode(const string& s) {
  array<int,256> d; d.fill(-1); for(int i=0;i<91;++i)d[(unsigned char)alphabet[i]]=i;
  vector<unsigned char> out; int value=-1,bits=0; unsigned int queue=0;
  for(unsigned char ch:s){int x=d[ch];if(x<0)continue;if(value<0)value=x;else{value+=x*91;queue|=(unsigned) value<<bits;bits+=(value&8191)>88?13:14;while(bits>=8){out.push_back(queue&255);queue>>=8;bits-=8;}value=-1;}}
  if(value>=0)out.push_back((queue|(value<<bits))&255); return out;
}
struct Snake { int n=0,m=0,c=0,steps=0,previous=-1,best=0; vector<int> desired,color; int food[MAX_N][MAX_N]{}; vector<pair<int,int>> position; };
int max_steps(const Snake&s){return MAX_STEPS_PER_CELL*s.n*s.n;}
int score(const Snake&s){int e=0;for(int i=0;i<(int)s.color.size();++i)e+=s.color[i]!=s.desired[i];return s.steps+10000*(e+2*(s.m-(int)s.color.size()));}
int food_count(const Snake&s){int z=0;for(int i=0;i<s.n;++i)for(int j=0;j<s.n;++j)z+=s.food[i][j]!=0;return z;}
bool complete(const Snake&s){if(food_count(s)!=0||s.position.size()!=static_cast<size_t>(s.m))return false;for(int i=0;i<s.m;++i)if(s.color[i]!=s.desired[i])return false;return true;}
array<bool,4> legal(const Snake&s){static int dr[4]={-1,1,0,0},dc[4]={0,0,-1,1};array<bool,4>a{};auto[hx,hy]=s.position[0];for(int k=0;k<4;++k){int r=hx+dr[k],c=hy+dc[k];a[k]=0<=r&&r<s.n&&0<=c&&c<s.n&&(s.position.size()<2||s.position[1]!=make_pair(r,c));}return a;}
struct Preview {int food=0,collision=0,length=0,error_delta=0;long long score_delta=0;bool food_match=false,body=false;};
Preview preview(const Snake&s,int action){Preview p;if(!legal(s)[action])return p;static int dr[4]={-1,1,0,0},dc[4]={0,0,-1,1};auto [r,c]=s.position[0];r+=dr[action];c+=dc[action];p.food=s.food[r][c];int old=s.position.size();p.length=p.food?old+1:old;if(!p.food)for(int h=1;h<=old-2;++h)if(s.position[h-1]==make_pair(r,c)){p.collision=h;p.length=h+1;break;}int olde=0,newe=0;for(int i=0;i<old;++i)olde+=s.color[i]!=s.desired[i];for(int i=0;i<p.length;++i){int col=i<old?s.color[i]:p.food;newe+=col!=s.desired[i];}p.error_delta=newe-olde;p.food_match=p.food&&old<s.m&&p.food==s.desired[old];p.body=p.collision!=0;long long after=s.steps+1+10000LL*(newe+2*(s.m-p.length));p.score_delta=after-score(s);return p;}
void encode(const Snake&s, vector<unsigned char>&food, vector<float>&bf, vector<unsigned char>&sc, vector<unsigned char>&sp, vector<float>&gf, vector<unsigned char>&prev, vector<unsigned char>&ac, vector<float>&af, vector<unsigned char>&mask){fill(food.begin(),food.end(),0);fill(bf.begin(),bf.end(),0);fill(sc.begin(),sc.end(),0);fill(sp.begin(),sp.end(),255);fill(gf.begin(),gf.end(),0);fill(prev.begin(),prev.end(),0);fill(ac.begin(),ac.end(),0);fill(af.begin(),af.end(),0);auto at=[&](int k,int r,int c)->float&{return bf[(k*16+r)*16+c];};for(int r=0;r<s.n;++r)for(int c=0;c<s.n;++c)food[r*16+c]=s.food[r][c];auto [hr,hc]=s.position[0];for(int r=0;r<s.n;++r)for(int c=0;c<s.n;++c){at(0,r,c)=1;at(5,r,c)=(r-hr)/15.0f;at(6,r,c)=(c-hc)/15.0f;at(7,r,c)=(abs(r-hr)+abs(c-hc))/30.0f;}for(int i=0;i<(int)s.position.size();++i){auto[r,c]=s.position[i];at(1,r,c)=1;at(2,r,c)=i==0;at(3,r,c)=i>0&&i+1<(int)s.position.size();at(4,r,c)=i+1==(int)s.position.size();sp[2*i]=r;sp[2*i+1]=c;}for(int i=0;i<s.m;++i)sc[2*i]=s.desired[i];for(int i=0;i<(int)s.color.size();++i)sc[2*i+1]=s.color[i];int len=s.position.size(),errors=0,prefix=0;for(int i=0;i<len;++i)errors+=s.color[i]!=s.desired[i];for(;prefix<len&&s.color[prefix]==s.desired[prefix];++prefix){}gf={s.n/16.0f,s.m/192.0f,s.c/7.0f,len/(float)max(s.m,1),food_count(s)/(float)max(s.m-5,1),s.steps/(float)max(max_steps(s),1),(max_steps(s)-s.steps)/(float)max(max_steps(s),1),errors/(float)max(s.m,1),prefix/(float)max(s.m,1),(score(s)-s.best)/(float)max(20000*s.m+max_steps(s),1)};if(s.previous>=0)prev[0]=s.previous+1;auto lm=legal(s);for(int k=0;k<4;++k){mask[k]=lm[k];auto p=preview(s,k);if(!lm[k])continue;ac[k]=p.food;int o=k*ACTION_FEATURES;af[o]=p.food_match;af[o+1]=p.body;af[o+2]=p.collision/(float)max(s.m,1);af[o+3]=p.length/(float)max(s.m,1);af[o+4]=(p.length-len)/(float)max(s.m,1);af[o+5]=p.error_delta/(float)max(s.m,1);af[o+6]=p.score_delta/(float)(20000*s.m+1);}}
void step(Snake&s,int action){static int dr[4]={-1,1,0,0},dc[4]={0,0,-1,1};auto oldp=s.position;auto oldc=s.color;pair<int,int> dest={oldp[0].first+dr[action],oldp[0].second+dc[action]};s.position.insert(s.position.begin(),dest);int f=s.food[dest.first][dest.second];if(f){s.food[dest.first][dest.second]=0;s.color.push_back(f);}else s.position.pop_back();for(int h=1;h+1<(int)s.position.size();++h)if(s.position[h]==dest){for(int q=h+1;q<(int)s.position.size();++q)s.food[s.position[q].first][s.position[q].second]=s.color[q];s.position.resize(h+1);s.color.resize(h+1);break;}++s.steps;s.previous=action;}
torch::jit::script::Module load(){auto b=decode(model_data);string d((char*)b.data(),b.size());istringstream in(d);return torch::jit::load(in,torch::kCPU);}
template<class T> torch::Tensor tensor(vector<T>&v,initializer_list<int64_t>shape,torch::ScalarType type){return torch::from_blob(v.data(),shape,torch::TensorOptions().dtype(type)).clone();}
}
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);at::set_num_threads(1);at::set_num_interop_threads(1);Snake s;if(!(cin>>s.n>>s.m>>s.c))return 0;s.desired.resize(s.m);for(int&x:s.desired)cin>>x;for(int r=0;r<s.n;++r)for(int c=0;c<s.n;++c)cin>>s.food[r][c];s.position.resize(5);s.color.assign(5,1);for(int i=0;i<5;++i)s.position[i]={4-i,0};s.best=score(s);auto module=load();torch::NoGradGuard guard;vector<unsigned char>food(256),sc(384),sp(384),prev(1),ac(4),mask(4);vector<float>bf(8*256),gf(10),af(28);while(s.steps<max_steps(s)&&!complete(s)){encode(s,food,bf,sc,sp,gf,prev,ac,af,mask);auto logits=module.forward({tensor(food,{1,16,16},torch::kUInt8),tensor(bf,{1,8,16,16},torch::kFloat32),tensor(sc,{1,192,2},torch::kUInt8),tensor(sp,{1,192,2},torch::kUInt8),tensor(gf,{1,10},torch::kFloat32),tensor(prev,{1,1},torch::kUInt8),tensor(ac,{1,4},torch::kUInt8),tensor(af,{1,4,7},torch::kFloat32),tensor(mask,{1,4},torch::kUInt8)}).toTensor();@ACTION_SELECTION@ if(action<0)break;cout<<"UDLR"[action]<<'\n';step(s,action);s.best=min(s.best,score(s));}return 0;}
"""

ARGMAX_ACTION_SELECTION = """auto lm=legal(s);int action=-1;float best=-numeric_limits<float>::infinity();for(int k=0;k<4;++k)if(lm[k]&&logits[0][k].item<float>()>best){best=logits[0][k].item<float>();action=k;}"""
SOFTMAX_ACTION_SELECTION = """auto lm=legal(s);array<double,4>w{};double total=0;for(int k=0;k<4;++k)if(lm[k]){w[k]=max(0.0,(double)logits[0][k].item<float>());total+=w[k];}int action=-1;if(total>0){static mt19937 rng(random_device{}());discrete_distribution<int>d(w.begin(),w.end());action=d(rng);}else for(int k=0;k<4;++k)if(lm[k]){action=k;break;}"""


def find_latest_run(artifact_dir: Path) -> Path:
    runs = [path for path in artifact_dir.glob("run_*") if (path / "checkpoint_latest.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"no completed PPO run found under {artifact_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--softmax", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir or find_latest_run(ROOT / "contests/ahc-063/artifacts/ppo")
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint = run_dir / "checkpoint_latest.pt"
    output = args.output or run_dir / "submit.cpp"
    model_bytes = export_torchscript(checkpoint, config, args.softmax)
    action_selection = SOFTMAX_ACTION_SELECTION if args.softmax else ARGMAX_ACTION_SELECTION
    source = CPP_TEMPLATE.replace("@MODEL@", c_string_chunks(base91_encode(model_bytes)))
    source = source.replace("@MAX_STEPS_PER_CELL@", str(int(config["max_steps_per_cell"])))
    output.write_text(source.replace("@ACTION_SELECTION@", action_selection))
    print(f"run_dir={run_dir}")
    print(f"checkpoint={checkpoint}")
    print(f"torchscript_bytes={len(model_bytes)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
