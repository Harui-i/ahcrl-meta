#include <ATen/Parallel.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

constexpr int N = 10;
constexpr int T = 100;
constexpr int MAX_PLAYERS = 8;
constexpr int MAX_LEVEL = 5;
constexpr int NUM_PLANES = 154;
constexpr int PF_PARTICLES = 16;
constexpr double ACTION_TEMPERATURE = 0.75;
constexpr int ORACLE_PARAMS_PER_PLAYER = 5;
constexpr int PLAYER_AGG_FEATURES = 4;
constexpr int PLANE_M = 24;
constexpr int PLANE_U = 25;
constexpr int PLANE_SCORE_RATIO = 26;
constexpr int PLANE_SCORE_DIFF = 27;
constexpr int PLANE_LEGAL_MASK = 28;
constexpr int PLANE_PLAYER_SCORE_START = 29;
constexpr int PLANE_ORACLE_PARAM_START = PLANE_PLAYER_SCORE_START + MAX_PLAYERS;
constexpr int PLANE_COMP_START = PLANE_ORACLE_PARAM_START + MAX_PLAYERS * ORACLE_PARAMS_PER_PLAYER;
constexpr int PLANE_REACH_START = PLANE_COMP_START + MAX_PLAYERS;
constexpr int PLANE_NEXT_GREEDY_START = PLANE_REACH_START + MAX_PLAYERS;
constexpr int PLANE_DIST_OWNER_START = PLANE_NEXT_GREEDY_START + MAX_PLAYERS;
constexpr int PLANE_DIST_COMP_START = PLANE_DIST_OWNER_START + MAX_PLAYERS;
constexpr int PLANE_DIST_CENTER = PLANE_DIST_COMP_START + MAX_PLAYERS;
constexpr int PLANE_X_NORM = PLANE_DIST_CENTER + 1;
constexpr int PLANE_Y_NORM = PLANE_X_NORM + 1;
constexpr int PLANE_POS0_X_NORM = PLANE_Y_NORM + 1;
constexpr int PLANE_POS0_Y_NORM = PLANE_POS0_X_NORM + 1;
constexpr int PLANE_PLAYER_AGG_START = PLANE_POS0_Y_NORM + 1;
constexpr int PLAYER_AGG_OWNER_LEVEL_SUM = 0;
constexpr int PLAYER_AGG_OWNER_LEVEL_VALUE_SUM = 1;
constexpr int PLAYER_AGG_COMP_LEVEL_SUM = 2;
constexpr int PLAYER_AGG_COMP_LEVEL_VALUE_SUM = 3;

const string kBase91Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\"";

const string kEncodedModel =
    "UX^FKm$GT20kAAuW]BXLvFd~RV8<&^3L=X:&p`eGyFJVFBk|X(p~8F;CkEl~VM%n}opB/?)|@$E@~]kCVx|7?(tF7drXuc9JHcs.uTN<zX&TcGinQtPji\"`s"
    "8^@b]|&v,<,.uC{Tn3n\"#(4[E`D\"2>URJINZ55)hQG4rZ2pA3(aLi_Otw(<VC_>{tu$hSQ!|It\"L1F}C<)kBTFEB<X:&BAIYr(cF<CJVhBwQBw^U$A;v#$OX"
    "j`3Iw({Ln`gt.N}=?eA8.ZL>f\"2}AMcGSt2FlNo>hq?op]3X<4Pq@JpP?s_C7(fAm+_~f_up;XSCpB.CuW1K9~ineu15WF<~^Xl1pAPWF5q?v~:tr3q/nIPY"
    "NLB`9~C&$J3L!~{(R2#^gtu}Z~0FovIA\"}:yg(3A=~0WTLl?^_8([2e_C_c|K>/>Rt,}\"j(^3Fj,01s`x|>pVV7BOAqu?X{>PuC>lWPGr1vk,=_nHx0k]KJJ"
    "_x#}r:GOtW#rQ@<FtW+Co>uAbuh_oLb#miGW]|!+2(oN9~5a;;W/Y4Qq8M@K|ub|O3`RSA[@[J1L8Mi6=Ij\"iX8G!yE{nYh~]E6rR~+He9g$muR]|siV[=^Q"
    "fA`@gN,|J|]iZ/VOMt6?n\"#(>&i~ty5F1M+2Rk%}cS&Bln+piLbL=s0,`~/>DtjE)A!~n(hNF?A|7aFtwQHL@TLAQDt_MV,`w{H!4}tQT_ItgNJ?ILA.\"aYC"
    "ryXwpWN?3n<)0Ac2xCu3)=OtC*eXq&.F,(tWl`1_try3+?CEfA@:{hfuhW`LYI6nxKR?_3WLuDMcynH.Tgq>`[BtA*l\"WumL@K;vb#AAaF`sQ@%AB*DAQtuW"
    "@)=QiqNBfLi\"Ith~xQ;C_sJCeAz,Y*#U7v`)[>c#%_4C0M..N1!(/Ce\"YA:_gtvWz?(AEcAA~C\"Cn?YLRAyWAsFkSVmG&_<~OWFV=Vj|7,tidF7ypqI@aF:v"
    "kB+>aFBt<)DABt\"X^LF`qyYV4AknAAm_5F/aU@(_OAQ)b4nmmodWcG*_{TK>2L8vt+P@\"FRt,(WL_K\"vmEW+P@sFh&EJ5FRt3rclkBDt4}7YI.lL;U4}l\"9u"
    ")tgGhtwugsB#K?cuxNH`6F7_M5B.UzKqn]m_ds[aFKIQOJ3WY=8=biTuWikhLPz&8t(^${=2w=(/(d$S~=xW9Hq35Wb@KF;Cq:1L)|#QU4hGry7BYN/B)kGB"
    "zYE_|nZa[Vg0ey[P*JB=aoFx$TeGVOp0:3=.Z4oS#AJ1AWJVZR<vA|*>Jcx\";}5gk3#(1(@PJvggGvv(J1A*8*~EmN[`BVF`)I>2eLG\"HY0w^Qk|wW/hQW?s"
    "[}FasQvAROo_^{<,C^a~,>%PCCm_Qw1T0MRLtG|{m(v{ht<~G,n\"`pY~*.@tu(5up@5s\"f$$K?Stn&$}u`0tJA7F<C9(4A6C#(Lj0E9vbnjJ5~<D/FFM$\"RY"
    "*5!Lk|>XsZSW~vEc$MbL%|!+FCAG~r2W<4{L}yiSGCO{HrSqV*8F2nu+[h1X71&B*@PcM56a4k1Fxeh)NN~(pmmTt[A;Sz?N8Tvdo/aBsdE?>CzCo(Gb[kN/"
    "%mjHA|1ZXjcSjth}|sjB,4:)NaH`Y1,;.MPX51^)cZSAryqiqK#tg$vsUYZCU/E)q?#~fY9}F_%{at4A|HWB{>~G4lkZBMq@LFn((AA\"!rlBAMAw1(oB3L<s"
    "VZ+>dAsBqW]K#ta&pB3F4FHAN?<sZt+>T@8s1W,AN53Y.A&_t(8VxR^\"7Y!A8XGCe\"L&i+}RHO6Ccs>cW|fAAGpvaq@@a^[I[FIvzRU|$(4*bGSn/oCC<>/~"
    "S&4}2Fh_#}DA;sMECCJ`Dt{}:j~F0_zy)5k~L1VZ]j\"EiRi0cln_dm*~h6vpTDS*!rwFhhyS:JQ{UnLY@1oWNDCYi6o`K<a&<$H?vAn?ULB\"]v%h`FRA=J~~"
    "5Fj#AAit9uz?[QcytB4M5F:C&alB:>X7(,[V;PxD&Xo1vKBwHA\"YDAJh&>zke@dqb^Y>t}X=MLB_tr\"X2/5c;)ng&>$\"$Y@KVL@Q]&$~]_J*`J#=>sBwtKtR"
    ":1^vj(;.D|Ep9K,^yDU/r=XFUw(XajTROqRt\"X]K}F~wj?k~Xq5}.h&A9_g=dG71,Wng}GBz7s5*0RFtite6d_a(@Tt*i~^n.}<)wXN4)uWj<I]n=~Y45~Ny"
    "zx$*^}KlF/rs#Ax1VNjBoF,$[`c=_FMBMV%~i/G1W?#Acx0t[Kc`~w@)P@^p&C66CBAqZ\")_\"~&,r=|})C8XF*fG?nTV<A^nuuoggG!v?$4}BH,yf+]io)m1"
    "~.h~D*GyN8kDYSY1JA_KciVZ[h2K/In(puxRtA2KF`o10BTjL`>^Jd[NM{=6muO>KBxDOZ$@Iho1Y}+&L\"{2HAM!jw$AJ4Cc8)VMZ1x}^)w`nFMKT7[B{k1%"
    "Y=dL<y4}9>8~*vuo[Sm_RtcB=VHBywKVoP%~E\"Qh5FjH>2`~`K3yq6,(X:n4vOqWxJQC[`ItZ{lF2([>/c~Cv(pKz/nq(,BVaS|3lgBJ]Q#\"rhfH#n<X=NwY"
    "6sa|mP@E7v2((ABtAA5F/CJV8A:CAA5FBA)AhqtB+>4FD\"]L~]gq<)q?hAlBn?DA3rAA=~0BAAPAyKC\"kBAA`h/$sMfFQqcZ(TL/51R)/>9}#\"GX,`kDaq6*"
    "u`eq@F`ZdLf\"C>fAeuDAjqtBbL+_zqADEAk|7,V&ZRStWZIAG7Y*6K2<Ar,Zo4[PH4lJD7xK>v3T.<V:reM%V2LQ0[u>JLH9aG9T#j!_9LDpvre~jL\"XRxQ*"
    "%q|W(C|*YL)UM~f\"P}\"jI>H`Ct8YqQA<?}\"X,VD_y,iW]QpqhF$@7WD|],m(pC34+}Qt7]kRwr&2/n1q]yj?%G8IK\"_FnFeW.@QXk|B&_X/?~AFE|jq>MGNZ"
    ".4yF^3QTh~K/Bwp(%KB`UtrC@@gHBq@FaiqW^km+@V;(.16aO&r@p1WODM>>+o%vVsxWCtpZl+>Put3o^)vQny=v:L/V:1i_$A5CRtr?bA@}zIa_VwId;vp`"
    "6I|22(/\"9W>>>V#G$}euTKU|))Gj&>ryY*!>Hc>v+}kBe\"7#*hxLQtN|Y~W~={J3.JQK{LL/,Am|8B$MH~KrUZ`~}F4IlBh~GBnIuu}VgABYMAQw6a(MOKSq"
    "D*34`EE_`vPSfFn!|D7,M`/v|}L)rA$(j)vEK1W+>Ll`0qoty3,G^_PZvfnHSD0ZIAPwBt2W(?Rt1uwkX~.F$$vY<OVw}+]>tJRtu}]ieG{L~wm(F\"C&h~XX"
    "DG,~Cry9MwF/=Auni#BCh#<Cr|R>}Fy|!W]qZL4sKq~Bh^rC+r=>A~kqsBfL;P=~8WFV6A&CI)UL7v&nhBh?JL{bi6lM{IeuHjYAoS^LN;v4]vns)aODUZX)"
    "BGvt1r^H`Lj7+Wdu@DPGCt1*E^sF`uOX&_5yuW|L9~:CRVAAx|eu1Wm.&[0}r}WFZC,}UJo_>~QVU@\"FDtR*$M$G=CuZuu`LADu+(A;v8BXL#HO2)#+Wuo77"
    "$Wd~yLE\"*V&_2[[sOL7~B|GDAAy|_)/>DAQD|Lv?:I^C)Mo\"1u#f??/y!(]q@QC\"tB{EywW+oZ{c\"i&a4MvR8tqyoBG.4InA5J:C}.1K6G\"~c+pK7^O|fAK{"
    "={I**V`Fht9r((f_T|n+~]ZM&nx&?@s?RA2[BMStfY>>cLj6$xz(~]m;rEHOlB=~=%J~fM{|&C0YiS%mdTXs\"R.I]FIhv/9~v}b4ZR,1Z:`1q98@4r0Y@K^>"
    "0)0*BG0C1Z_)O`y[WraUdLDt9W(TCA2ZNJd~7IKtU4#>Q\"Pg?RoICtRCbL+H/zo%#\"g+><pQr~%aSqT/R|V!4}&|7FQ*83hA:`z3G`nL,}FCdBsAxhx?Twbc"
    "~~(A/$HYWK+kcSPk$M03+u{vBB1|9+quXR,[2uLv<Q3I]XJ5CM:{c+5KzF&KWZgsVXEE7aGK\"Y7y.TXXj~mO@TUE..EnWx1pC\"XtM)cR~O`)6i)G0AxU1X!t"
    "Z\"{KStmZOqcAHA5=BG=,n?8Fmz*uY4yWk\"XX)_iq.(iX{K1|~X%*vKmn_,/>u?SwfWW~q~PDLc]qKQ]67C)MiG\"y/aq~%GqFBtd~kS8ypSQ)e~A5}A_:?A.5"
    "WF7CFDoM\"Mfz;R[)0PZCqd/A316,oBe_{|A.R&zc0[uWG>yFct/o|~L_c|9)It|FqFMnUhgAK&V~$/H?Z|E>2LjnA&)&jV\"NpVb?OJ!s{GKOAHl\"&>aFf_.d"
    "3K>}cC~*4hh^FoP^I@bGzn:vp=%\"4}!3.>sCJ&2?I`4FzS6WG\"^XM@*_j|\"X2K,`Uo.C~~:?l|3WVNL?rFHDZ~fAn(5K!\"tWr?^K<sAA$_7C<)#Ai|0BO>^Q"
    "mDAwIM8Fz_Rqg=M?g|t(?A!~!rKxFs~vKH$AQDWZ)M}Kj|;vDM3Fn_#W,M|Frv<v=Vg_1k+.\"jeGB\"UZR~|+YFW/Yxr!k_>}rAczC&EtypCziE_A7F8)M){F"
    "qF|(23qH2|a#=&L_hqqSu+r_%sR#4gPR24%|NC`/vq#z:Hg>8Fqaf)WFwqZV|XgA]Cri~F&|kn{LH?t{U+Wvx?@+5$AM2Ec2euE@1QEwa|;49G6F&}M5TK<s"
    "?Q@@o?rcRYE[QFGog(hN8Mft7x0[QDHFXYd+@VF|y,Y4\"FX7{oH^/>(_Y&#}j\"VZXX7LEI,TiLOcJFdxuKt/8~Jt+fAG8v(XAAg|3(_VXL/CIAq.~CK&iL.>"
    "~yeWt[*_wtW&r?;i>pO+NC9~}C^vxKL?U_^)xWEA6aiyKP<yk0%3zR%wg(/hf=}LrC0Mb~pyT&Q>k?):y8n3u(/F+~Vew){0N&>&Z!`nh$OX3L=v:a^fxX.n"
    "exNh!SnI;yk4nGc1{TYL1L8{k_F;!~7Fl#n(gG>y%(bsjRF9q:wB9~7v`ryi:Wyw#T8}>/7JpFSxGTnh|rQAqyb|N5!~%nS&{LlBAw;s8}2FcJTHU7VL?mYV"
    "GC]E~CRAh=ADQYI5IN87Pe5WCGx\"v??XOCR|P)(YUJ@(g~eFFodZ0MbFxknrT4C~eZ$a_)I`i_tr8M4Gj\"7fV)9s5aY@$F0>%a;LZRA\"x}DME\"?}\"Xj?8vHA"
    "HBwt<shBk\"a|aX._=s(vQhbFT|2Wr?tQ6CnW2Kk_gtcZv?\"FD\"}~u/%\"?A/F;X8A}yhS2KcGpwkWDM_}Y1tu/>6LWO+B]>Z~N<+rXL%~CA<AOAEAD\"2?,_;v"
    "n(iLcGXI6,;0=Pr1`rGC7F]~sZyWVLrC1u&f$\"Zqk%vQ9qM//V,?\"yMEc4v`!~~C`ZA\"51tBn?PCTLX+uiZF;C4+Sv5d8R{)kBO)YLN|qLuEAGHZ=NqWu?zZ"
    "5V!Gc1RA0:^H@9/&@?KIkB(>@Lu~<~NjJ`g\"_sj`%quBZZ=^EOqak~%>H1v(VqI@F}4a>3#_MD_)/A:j7C;2c)~vqaL@tWa1tuY)%_1nKVuK=PRqx`*>p.GL"
    "gA1L(|SVhZ:>w\"_`4LqIv}sNVL4F+pOOcAuW`$*>mO/TlNr?l3hx:Trjx>q&N5yD03qqj@VQE\"vwFG<#mh!(g^#qyf+OrR!~luEtcX{y8B(Am_ebFVN?:CS*"
    ".h(^>A)vl_|\",H!AMZz,CF!1NEi&dAMZEO?/<{#(b@I`1LmBr?{FP?muG>VS6C.rtBlB\"vW+U~@P+1A*EM5Fqy$TAMAG.1B*1WXLqydBh=j_=prv0*6|4vf("
    "|4$GjQQtIAvAJh6KPD3r^L3R<vQV!>UW?{1/MBAGFqAYbXbF;y\")nB*VPA$}aLSqUZ/Vi>sF>rJVYM:4gCUE=`0qu_og(Byt6CmWA\"yn3rn?|EryKVt*tW:1"
    "St.AytmW$A=~k+hB?(hq`)tW~~;jZ\"!GOw%}*>o/\"CE)|7C2QeAPEA\"s2(B>A?DnA&qr4cyJQbjL|RxmU/l]%NiLO/V]xc3LoyT??:I_]CB}i`MskZFkzLl\""
    "z}QLEUIYs3{Eknn(eX:>[q3`eXP?MGBATLa1ZqW+4LwA/V4}vQ}}c4O?Pz$vZ%IN=~Q#=&K`/kBYM)G`Xkx:J2+L.CK\"H(wkNm[#h&^1tHsP3R0tIYxi%~W_"
    "&y%h)_:y]vyKDBsv?YtK7^<yv(nU`RI1xa9[>/H}$(ngiActzrxE7mZ\"AMAGxS_vQ?ZL?R_ABGlx]j0QywNxE}%AF^V>WX~F(Xx=vCJ]vup~BmxAQMlF`h`)"
    "cNo_Ct7y>,1;T__zf)?!:CzSPX~Fvw7v}Jq>\"~$}pW`?\"v=%~J?VPAE@j>Aw0+_Vj^!~Mx{L|FZF<Uc_udkBzyZU&&@3TEhBrVM|8)@~T@xtcu.[y/@~z|3r"
    "bFm7<)D@~E$|l+ZB(=vA/V3FBn2(@VXF(k_X5*4~*[XI,M_K/FPW1*5Ghqt+f4qQ;Zfu7^s=11Oz~=%_rCQ*ms!_D|Lq*h3KrI*~rs+?jD,$m6T)\"CK(?|<?"
    ":Cr|8<<?%|=~DY+_~vkB9}O?eDzX,AXi1o*qnc44uZNZ})RA[hX/eFnZ1M;V/C6CmW;B?tR*KX1(s@%RvA~C4}lB3L7C|bTX9~@~IA5Lp1IA?PpFCt0MxL{>"
    "1N*SWTT5u0<T!GqFvWd~2R.yPAk\".C4AryuWlB)_\"~[aXX~RT\"`qH`gD!2HX4A:as=p>j\"G>)~W}xS2WQW!v`%1u.M{U<sJ]7U#CTZ|Lt?FL6&lNm\"wyaj?P"
    "LLFbOj9~}FItB%_Ep`tuh=/AAwfUcA7vjgK36gF/4Y7F#6vr3A_t:aFh~RDDQAz(Bt0v8}|MSwIbIA(t*BX|(O*s[QwY!~E_+TZ=GBY`j|bg[9y|]CTLp?Pq"
    "<~.}S/+uX+7wGBaFQ*pKQ)PDLSr(TKsF@$@VqcN?9W:>aFh|kZ/V3F~Cw(g~cF~yIAVLayvr$A&|AA7FpF/C4M~Fw$u>u7{Fk|sxWLi\"AVyWiA2WuWL?Qw>("
    "5K/`I4BVkU6A_)2(SL8vAYc4Bs@6kEKL/?([v`(<#>6~UB^)!~z[2(LAPAXL]KPA:>l_R\"d~4LB\"tB0Al_lZCCUQRt#}eXm_5<[sL7j__|9sNZ>QO2/g(?7}"
    "M<1+gsb=e~q,8hEGdr2uHY~L3koWr?PW$A|XlhCt#}AM9FSnTx%*R\"It|XR?tv4y~=QRdtg}vM!]310B#XOQlnM|V&DGPJM<?<I`^_1(gJV/~p3WdB#^.I$F"
    "vbJHwvku0*F_k|{T`Z$>z_6)GMtBkt_C?&WFJ1%,Ovv@ev9T{>}~/y5y>agAG+j49\"?}nUO{]h5a%h)_wnvZuBaG\"yR|lSJ?AtluDYI>eDn}]XlHNvOpb4m~"
    "{n)RIAXs;a#}<V4Fo}J*VK@NNBGCTRY}x#bPdRjtYV9tQ{2_qy$@jcJifVwfeMQ2GpGj.WR?&CMA$9\"XZj^KNwB\"=JUMuykB$<]/sshVJCs.6IeW*}9Gxn3T"
    "gBk]ec!_*h7LBwz,z?zK2IaV.@fGWt!(<@v>$pSt|([QAw,}COp\"=}`JZR%n_U\";qj<{4FoUXFoFlBv(aLG_eupiDB:FB\"{LQQI]tuw}K`Dq]`q?{9\"v_sM<"
    "D\"OB+>D\"PA}E2[&,.hiS!JP*|4SK{H9~bLaGG4j|[&kBqFF~HAmI2Ww*gF9yur_@p8P>,]oNp`Ly{}G>6F~FIwY=T!)[8SMMN`4CZr7MOcSqyai{5<BG.d:a"
    "%^}s?CU)@Qz@zWqK(~!sc#a|6~3FT*#}L_tAgUCM6C.2]X/Q1O!up+0|+hGYh[ZG\"vFZbLcAmub@5QTnWuZN|F<~(X7fL?\"yZ&=O0E9<gW}@5AX/o9)GD|WW"
    "1K&_7s7)o=YFj|_%1KZF)[f(8}9_Bt:vMhsgE%vE6KQEu|zXL|TK|FDui?8~~m`)vseGBz3(v?cLx\"[VaGdyX+xW$\"B*2?fG;v\"XNV7FS\"8h4,#uItAACwW+"
    "B\"I?Pt!r>>,>qFfWuW7~0_bB4}*>Sn4}u?cFrFBVAAPA)Arvv(qW~FAt^)Vq\"\"MU8r;Pit|DnXL`;r}r:&c~~I3$zMpBalUx`)N@aI$}U)s?FtRtRC`F7?Jw"
    "sBAM:33(mKyQNIu($M)A&F]~}J&|*:/>>`Qt.T|XE_X4BA%~N2RV.AM2Q(gNl`a`xV15f\"WB*Vf~NI7,5K~L?pBAt?F_iqFA?|6aQ@NWUna1p[)^w|Jt=JK@"
    "+0rE]X#^Tt!W`~!F4F8WQFNPqCb#oN).h\"RC6MJLzx$[k=#<Fx?40DNwvrPLH`y_OW;X(=hq0vBW&_fz*r)<jA3(SXWQ|[MVpNmAAYAYBFU|`si+aL\"v[C@)"
    "AFzn_)b)]R={.(*>\"RZL,uNh|FuF8vOjAG=~HD$A;vmZuWgG:F^CbXW)PDuo{4BGVq$$%hC_xq^)y?!~yAMsZ~syA\"A*2FDtJVRJ=Px|Aw2rY~Dt\"X;@AFEo"
    "CSt5\"~CqAtJV?F;sUZTLl/=v&5]|(Avr}NJB,xSSZZEAAtBhaM:v>$SX3~l[u(8Y5F;y2T6&TQ)\":aS.mI:6LAC|hFmiN?n7E<7P%~,xG{*>9GB|4}AM>QDt"
    "R\"{E8sNEWi?Pj|0_VC{Kr>5}qSh=1y0B_4QV,Fc%)VQWrG)v5%6~f\"#(@(k||.23C_hD<,S6QDBqZ||jNX,k.$jY_K>IGYQlRLgA.@d:r1cr:^<Vep5n^@ZM"
    "0OTa&>i=8^cB/$bHtAS7|Rl6<W}@bMDqAY!flhE|jBHCEC#mORP|l\"\"v+j5F|IEU1Masd4sxxNiG<v5a6KXFxq;vXLnbS\"[hr`g/DHpNiod~/}B[8FV>%X0<"
    "b_%=hvDY(Hpijd$^RJYIQ&jg2AzPQ)TW5F6yhqI>^r|9T4jWYC8rY<}DeyT:c~^EbGCV.M(b*|eBtKj^zKO%n?zQ.IFxv;+>2xfW0}QB.1Y*quI.O<%ajD,`"
    "_[:EskRL&qsZn4L>PALvbL+63(3AOD3r;VJ`xqRA4QQD{s2WfF|1X+}41E31|W55wG${T+:Vu@`3i#Q@p_OEJY3TKW`6)p%*8`=I<CQ@h\"h&=JQ{3_A*tKtP"
    "J}]XYs9L|F[};LsQ1[IY&>C~,4_(f~PW2[=eQ)O?WLnVZs0W2q[Q`~L@3LU#.=,.71eA5RRAo?u?G7KY$AjHLU^XWeB|7_/hkRHF~Cv?)Fft78`~xKMD],3w"
    "u(lt:POOO{cCy#l+6RKCRA1(+_4TDL).psm}<J5Fwq6FJhsW=v#TR@dM|3A*}VcG5yQ*2YHp4Le0`=]|mnR&Z~w`+L\"UV)bMQ\"m6F_71*~~+k_Vr/)Z~k=YC"
    "rngdR9Ow2TN)K?x|0v(}?K3ya|L@lcK]\"VQtNWTqgB;)^@m}*$=h:A=o3(Lc2;#}+)4Fy_6vn?~}<sAYn?zW/!\"(CKA\"cF*)*>eFODuBpB//\"s_~u?B_wqa&"
    "j)l_7sB\"ZC3Lqy/ChBk_3F]v?Lq_BtD||Mh~ds/CZ~|Kgt[Cu3r`h\"T)CHg\"vU;?nCQx#AtCD&zUbF?w^)/>M?/I.$Y)LQkuQVj?F_ryMxKXZFBtAAAA^X,="
    "~grCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlBAASqAAk_6Cn}0p#_~sB}z}2F%|d+5KE_AA+>DAAAeAIAC\"dBAABt]XyKXLiqRV=VDB"
    "IyuWuuti=1aS4r#.,rBwvMxJZ>VUl%ZKSt^,hL7Lo4UBdx+_`0;BgNkoFq#W|;HAk,MZGh_|Uq/&7M&3FUqi$\"_Z2L;bW]eq]XsQZ7&CgUYS!A^Xa]@~[QMx"
    "XQU4c|d=X:T2SEIAqIH{Al3L/h^XpM+GNzxP1BbFK12+~&L\"zZeL0:]6o&%*!~24r#AM3~Zu{$j?+`(t98>X!}&t`U,CVFF}H!{&_RDqZqh=TQx|\"vqi^W2["
    "J\"dMTqkB:&F~D}hV15dW#k\"X+?1:&q~XS`i`p1W+N~]Xt~p#_4}L?nqyL2gA_)y(cFV|lWuKdFpF~XQ)i~qCK&uWgA^)wk<VnI#Wg)I^$v3bNJDBAw\")DMZF"
    "I`lBM@*\"Jt@V}E;s1uiLwQz_iyL@!U9?tW$<dAAAk_/CVByK2KT|kBXL&_T|MEn?aFRt1W:>Z!E7FBIgjB}CduGO+^fwl_~ZzQ@n$}Z]!F&[aVMLA\"n[|Dxs"
    "7<!qtu(;H{8Z+~HAi_LcMV&>:CT|<J~LQq;X}s1FsHn`qwLB)_n+TAWyQ*<Vg\"n(+>_FLGNuw}xQl\"})|}@{2us=L.m_OmTXf~C\"7TkGuAbEZFD_:ap=p_cM"
    ",rc)I?MFt+m|bK4C]6GTw@m\"Ih}!bLWA!Ao(GX.>twtWC&!FBwi|eL+_jnj#$A;y&a`BZFlnmW@@{Lo1eWn?6LaF_s6W(?7Feu@)=KRtm+2(AHhq7C]>^K9v"
    "+($BL`hyC&^jU)|hLkO4UQy|ZV[JVFBtvr:OL?y|dBG>~K.CI&8AAt2WK>xQQqh:m6iH+\"!K4K_H_)=>YFznBA~FBw3(<h$AP+iLi\"_);AjquB8APDK&EA\"v"
    ",r8A)_DB6KYK#~iq})&h]zMU^|4L2FZVY@gL.1tWXX.G+nPc=KhR:C/`q37FStCt0Av~R\"E\"XtW40KFq}T_sqi8zMug6:(uz}bxJjBxq5r2?eB^KnW3K&>o["
    "R&,T]Ezt8$M@W`bypq@<5MYnPpwY^F@3X4_2uWEDMEY@Q/pC(,?|^cUtrn?XTXk_bvl6qQ)3ZF8A+[QYg?0R?mh@hN0M&ATL5FHl,}D<8LU%B1r(_}\"s7,kw"
    "H_=~N%9Vj`usvTyB]Q^qkBS]IB}0UZ&a&V_[RV=>AFmF<Uq+\"~|iuulWj?awjUg4={+H>GY==Q@^=~sN[VPAbL1^6<kBg~H_RtQtRO7~dtE|aLk_rfmZ,<c+"
    ":Feuf(cFAD[a3(zQf\"ks{E*0_sd~bK(n#u.J&_nyiS65>Pi\"=NZLfAM@fGlkQA]KYIjSE<6F\"v0B5hKQ$^@1l*m`31WB/ZKW*>d+?v?U)k+u_~xQh~W+=vl`"
    "cvKqJCo__[pyl(q/L~=9(A(;M$q?d\"E+<)hAr|1K+>T\"aL#^OqT|QJf^/F}wj?6Lry$}/>F\"B*pu7R:4^)OOeGOAAM{E<vItU%M3=jZtGO/BAwm+jU(=o>n("
    "v<b}tG7vdSu>RI!B:jH?3nh*%>iF};QVlBM@dG:F$)gLjn0ZQ4XFKL?D?@]L]nIZB5~EtveYr?;QqF#TAAx|_syKAA4r2K9~i|]X>>6F\"vo(AA=vcB~=|Kgt"
    "Rt<)2FoFBAm_)ISlq($^!q(XEY^FTtHA~F!pUEpN1L51tZtNhHl_#(|L!AK&_46FFrP(|X3L0|YA$stDwW8M5Rq1HYbL+_9vJtd~[KhqeuFCc}S|`}oAx^tu"
    "IhVL2[AAVFush&\"L(bN#NflZnHRDn/tseSKL9ZiNf~HJ?W>3DGtGR\"5K4C+}q}./fw`!j3\"?Eq!73(y/V>N/#Y=PpTkWRC=B]3DSmW3(o.7`WLLPOG5ScUkG"
    "17htYkyW!mj,[5+`/yFY,AEu2Wc)!Lg2^}z~\"J__rq_N+_)HP+UhP;f||r\"O/`eD;[KkV*n_i&zg6}\"v4(cZmU/?WII3)_Ft^)8V2Yj64FtW@QGyi#+CGiRw"
    "p:=/IV4F/v}4~}|[Yc]X1F^3P%c4kB]6]_:v{ASSiSsKe~DsqW1L,Fz,1W]KKDbCFhq_zE4uQJW@8qGG5>s/Fn/`i?R)(n~wS>C~ai|{NJ3F8|3un?ANm\"[J"
    "^RFOCp/he\"SHW>zQDqvr_BXL#H|o0Y.d)[>T,Xv?fp(|xB{)Wq+T3s((t~Bw]4SlF,B\"|Z<Ipf7|_~&~;F[s8Ait{bFx4;.iM&4<~KDwiF85._D|6qnu<Vsy"
    "wV83xQgDt(X|>^GrMRj(.`V[u(<Jr,^HEmN&fG6~AV$@,_:F>.pqRP?{&a\"iVLZiV|hBTR1nsZY)$FH17vd=+=e^29Mhk\"T|oi2|:~ZV$M]W,1it@|R4o>GD"
    "mNkAOu?)>X=4!Z4M*^_\"pB}EFk8sqiJBoL\"XPX64^0ZAm_RzF!,XA~oI_X~V.`U4]v*V!\"(X<@|E2L#%w}q`Pw1u(Ax\"I@fA<)5K5FBtA*q?BAuWEATq3rAA"
    "6C_)1K~F=~VZAA6CAA]KWql~u?BABA&_D|>T8Aht,}kBi\"AAC\"e+eL1LznAY^L9FCAAM1LAt;CNV\"]~rdZz?Q/}nKtM$]W(67a`/O@byIV:>cFi|mW)A4F3r"
    "2KH`Awqa=VAAAACA3rv?^KAAiu]o&GLto~}L9v%C?LwC4R2+DMLBgDQAAA1W$}BG5Fa|0*B\";CuuNVaGOwA*kBH/__/=~2vci_V+nkT3kDs8ok/bbC/}u{8X"
    "5ymT|LsK(k&aJtZAC*p%[Q/[~&b@PW&npd.@MBbzmW!Wpbht+rp=bM&q>G7M&E}v8C}@6FOw+(~N1J:CgVQ4C\"J*^MeM3C<vFOh`<[T0V6J:]sF%3s\"GXqV{"
    "Kv_JF\"NJ>QSzMEN~x)iwTc$}U`iqe$#Y%h=C)v.hr%34<)t%.Bsw*)P@d\"eW6h_Q,h1XH*zE@qmB;L_Q=vJd(3LdDipy[q\"AppXWg~aF}F.$<)(G$_#r%hl`"
    "5vG%pWv(ak6aBwP[HL<Z2?l0!1#r(fDBE\"~B0K=whq}4=_/C)vDMVFVcyyMV[/!vHYFh5F&thdtB_Q&q\"CXLdLV|AYr?n_RA=JyKY?haj|ULl1K&=JjhAwi|"
    "vYn`nFs|kU{EAwFB2WJ?PAZ2\"A?~h&h6%GiqKcMJ&E:sf(_4o?`BUx3}K`&nlu`ql%E4GBTLBV_1`UR%+?7CsZ#r@K/yinG&B.I8cqO7*U{s_%zsCGV\"Ak\"M"
    "cvQgBJ;RmFa1f4vPbFd+%VYG6&]a=&YGvtc&Z6FGRqtZ<@n_YFQ^T@EBD|QA_Ejt4}?<j_it~C=JC_Eq&dV~n_PECcbLm_!tbZ8M2Q.sJtPXX*PA2KEG\"sAw"
    "r?wVSn2uyKk_;~xrE@aFVLU|3<,\"U0$5hG<s3rkk4>L{Or5knHWw^C0(bLRGA*TXE\"Vx<@WLvwIt@V9~|k&C6{FOytdusBK@xnaqk%l_cv(~pWp?7vgq8ARt"
    "_)sB7FjtIA$A;CyW7FfwQ*lB7FBAbL~FAAXLcGSA:>(_}FIA`F7vB[)eLEPw>Tv?j\"<)7}s/ZyIt5KVLz_cZTLkAuW8A;v2W~J!AmW^LAAJVEA!F7BQEQcff"
    "bHFZNYG9y9L~Do1kx}9>}L+4%C1+SLzqvT83q=3I=)T)Y)jqmD>+A;Ed`G!WMA}A0QZG5r/0aNiw4(B\"UB):fg+K#5Azy&rwSR/v\"}P)w)d?@.$MKQF`}$BK"
    "aM(_Az_=fG4yOZ:>G^_0z0:TL/k_oW~){/_Iv$d2h`XOS|y?cLcDkamW{KmFV+HMzL!|&,|4@POz&a9VxF>~em<hM@l|A^\"#>y%~e(P@{FbCY}5t0RH4Ytjs"
    "IW%qF/aL&\"&C[VAGQA]2ZMotME.ADtAAq\"nuHAE\"=VBH\"C)X1*6L6Ca&+>j\"J*Bt9Q5F4(HM8MRt)viL5Lht!W$M{KgA2Wi\"2WXLk\"lBP?,Sm;AQ*lTLJ]|z"
    ".4p_Tt1u2rfFKIM|.t0Elnv(sBq>LyxFjgd^I`hSx*^E}y.9fsXFaylWTLA\"gt=%[>[:y|)v.)khYF_v6*6,=sS@RJ$_PzDcIhf^.Un0uxj?CqxywsL?ikR&"
    "fgh~<vfu]&,>A\"At4}&F\"CvuDASD`Up=u?8CZ&E@HAOxg~|E(t,}C3$_vsN!5idFQwEp<4oB8v7B5hP?gwF|}=o,.L[|q?ZGcv.}^LqKQz@o2KZF\"C^)yKBH"
    ".eCtc)4Ly|eW+>S?Ttd+sB_EqCaq+>#AJVv?7y<~`s<=^K<sv(q(g~:ymZ7?8F&qKV)A;C!ruWL?qC1Bd~{LjnB*aL7FpyF!FK>gb4/(4A_q|b$h*~PA`hj/"
    "{EnDmi$MSzBw|X7LoLJ*fLQ{pIa*AADwC*/h]W:CmuAM1~Bq;R=JL/ATYAm_$q5}+T7RFucc!5;Q2n,W=MvdynN<d=/_/y$C33F_ND2B=~q/k|G!SC?LMD;X"
    "sB.C^HMElN_QZFX+v?*_t1,25>y`ZW@pcIF/t4@.^.x@<CjVUN)iml,{LX_S}pa0W+T?|ecq}4cFzT2WMQXX/.!W|(+hIlaqbs~(_05yV>*_cssaZ~ABgta&"
    "j?VQzn(X+>J{iqO+$[NSXFUBHj`FbC/C3f=?gw)[A*6~/I?}9KL`x|F{+>F\"l+<V%~!~Y&DYgAeu~e_Kiq!r>>&^pyItuuHAS#3rBBmIs8s1&\"bX)M3Ln1<X"
    "9hWWBtsZ|L8GqF*W*5JGrvIAo_/voByW+_#2tuI5P?AD),uKyK3INx?),\"9W[~HAIVqWQQ*kuW`VkSrOmc6uLH7LQZRk*nMMnx!m+Ge2^CtZdLsI+B9<>Pxq"
    "#$]>2RHoX+9aC\"PYkg1(!zYtzr|LdDWWSC@QPA@XVLL1Y&q{_FvT))=h*?nn,~7k`Q%_:XsNw>>~$aIA5DSqhl0W.h&#0AacSqcU]K$qyy/CCAC#f%D2=Q^y"
    "r42^VeeJF5M\"GmC?g/u~`BY@@LkI^]tBF?BA{ucGi~3V&HYMdIUWdN#F4Lh&kZFhDB3rWukB+0PwgN+>Vqv+8t=>+ka@d>P)7GEs@@IcuQ.o$XQ:{sI{rUJ`"
    "Az+(zU=?;c+5.J_EftI\"To0Q&c45.!,_+(YgvF@s%WpBE_bcc,ti4^$^cBU4wcfm{!>;dM813rBMAMBDdZ$ACD!WKCQ@8y8,QhjH~vBtmxH%1_o}TM?DYCXx"
    "[2kW)t=J<tE_ynH+vU\">sC<)\"L`FII9(iL3F]_!(fs+_hts)y?8FRtduDM|Li|kBK&PB^HCt8AcyF!J2{[|[KA6Al(z3&?j|3`xKA\":{{GBVQQxqcZ*5/&Rq"
    "pVv3NWAq>%|4bQgtHY/hh~St:9@J!GSt$$4~+~ynE&*2GXj24u40`:LLjEEtZM&|=Gk)h.%|p&Q@w?Sw`pH|SLnvTB33u?IfD\"VKdydZEM/`}ysB.vgzBt!%"
    "cCwQ#mB.{x%b;CpAgGB7.rY~3K+KYFcVqA{b>2vXiqkE?emFAA\"CxR(|JVv7G\"_ZAAl6gq|q!EeDR\";?}y3rx*M`QAi6F`y|QtZ~hAg}b@m?=s:C0Y$G`}kv"
    "t3yKht>b+>,_ODlB:>CHy_`)n?^QeAc46L)Kfu)g9~uJOca|3EK`4S#iYSfG[.]2]QDz:)OvBA}DM@B\"]EVZO6I=EygVB5Q?m1jqIA34$F331:MJ~slZv?o1"
    "\"`>L%BOJG2i+Rdo`IYy7TWX[r#6^_;SQjECKsETnBs@shFYli:SA0|%aD3D\"5ChW2:rvPDcL>KBD9Wog0E=sZF!>WLLy]~L,gS=pr:`VM.VOB&bLR?!{GDPX"
    "4K5yudnMj/_F2W.CV/et,}4Bg.RI`T@4wL+_bR$AN`atnD9~jkYS5t}~^E^vZ<L:n_I*k~#A:Fj?VQAGb/<)A`py@}BOz/tCC01K]Q2[yFXL3F7vv(DAAAyK"
    "gAAYAAAA+>fA3((AAAAA;v+r$AAAAA~CAAi\"HAC\"]XbLi\"HAAAJV|LEiRz0Vmi}3wxtx^La{F18,m|@F,|/$?ApIKq/W/?X4,TTXP/pFWZ_@kb2},_*Wx?Iu"
    "tZWVZFm}iqQob*C|u~8A:CIAZFQA8APA|L)_BtPA9~RtlB4MgAlBEACtYVAM#Rkqnu>CH?k|^X=J?LAAPLd#L2AV+>|E&_N[]2]PMC]Pz?RW0NmuW+v>AE*x"
    "oU{KJL?D^XA_&kzXTX{;}Fpc]CF`{k)P5K7~z\"kU\"~xq#1u(i?6/7v6r*/2qn/PAGvoVy>~E4\"X0J@:L78wY@Q;G.;7AwN|dA@*oRnk|KTlb:ETZ/tnBxH?r"
    "`]:=3[0xFxQ]5FwVwM:>OJ.Fr?2FtCTV@)9G1_&,z}M`WLPuq[zQmrdWpU!ah|+W7MM{Owad9VlBEtZW8$Rc6C%T?X]E[{ya0}nA$\"V;Qq=sd~UL1qUu8Abi"
    "V07Y3XFn3(XLi_5yl+LAPtZt<@cFt~YtZeP?kn>bLA>vNwNhp/qCtB_)*_wDH(*du:nILS4AdJK&Dv9Gf9sn=h%7[xL|jX\"A=CB*5KFBjtxa1[XQ;yh\"aGQ5"
    "1W?X.\"pq9[@hns&`MhE>lIl~&>eLmF`)lW4:!s8_E[n_AAnWl`_N_GoU.>[n{r`=;&;y^)4*VEF\"`>{F7vP&;7PL&_O+=>H_i\"]>WX2~.op6Hc=plu_J|Fz|"
    "xy85BG2_t0i]g~%^D0\"XmSfp?`H4`Kiw0Z$t\"~dAA5*_vf.bYZ{AGxU)ub;v4B!KBBAqou5YbAW_H5n_xn,}_B#~jqYW\"+&^ayFY#TrPaCTc(s8Q:CCf`LBH"
    "ZflapWXLPAAArvtB^Lk_:vtuaX1LT|e+@JCAB*[>F\"<sd~6FTq3(<VAGgtsBr?>/wZ/6)APq,$DArCIA{K8vCUP>8F;I[:f33FY11W<4,_@|3}>&~ET|o$0$"
    "{Qs~GpD3gLL`,bF0#~#~a#S#3KMv`{WLFHy|Qtv(1Q.F@}*>7AnT{]m^SqPx(Mm_J]lZtu3RY1\")NA9I5+AM._[s/v>TOcy^pX3};.B03uJOUSXC*U_g6}k|"
    "2rYB;Vft*,QJvLss[$C?hA(CR~&?jq;aGC/?[^x|Z&dG:y)v,v5L=~`o+&!L]_YYN&M?/n5$YVx/$Jf$CHuK]jn}]6YL{C[$u}`EQz9(E@IBSv]XBC^F#|=}"
    "~>F.X`j|5WJb7I+~2r@JJF`r>qZRGFbc/>3K0neYGCYev?q)e)y/2hmu,@H@hwt_*>DGSG<ZF0W:.FFU4AizAw|~vccs%CPC0Q#`,(At4Lgt+r||0/@n8uL$"
    "mbUt9)88M`4_EX]Xi>0w5qk4iEK<TtAtIc$tVujPnF5vhtFtn>K4d[]CigF`R*^4)=Tn@`OX;(Xleu^LaGQ\"mWEAMZ8A\"v(vk~K?5vItAAwq7,{X~F9v8BfL"
    "3LzqKtiLHBT|Jtn?G`>~Jzlx7;&ed$FjM36~H&~(M/qsUZOLv((0?T,v^K>y9B61~S<wIDCizL^^bS3ARnov2r}^V_<)|;I/Ml#oD7N\"*%_&AYSqbcF&Ib1n"
    "5,KaU:DnDcxYLR*_&CNVrB9v%`ZNG_Y_6a9t5FX7UcID1Xk_@yXXVdP<n}s*hS^3!>?Tn?Nz3B%tS_c1yPD7lhfA2>4~BwXt7YjB.h~w6?cF|7ac[qSK4FlZ"
    "D33^?srSi6NX2Fzqs1v(3y^CpBE_sFyvHAU7/C%V7F>sR*T|aFwAtYMW{e$r2K_K\"suWBh)_+0/Y,z{~VoZV&fk.1IAYr(IB;vdBXLAARVLLCAS&%hRFhA@)"
    "BAaqj?BG.7sEJ0e=l.bB*JC_!sa&eiI\"zyM@aFSqkB`JeA(Xv?^KQtAAH`;vAA&_:CAAAAAAk_RA8ABw_)3?{F;s#T.Vl8bt#}\"LE\"p&!K@K5FVBK>6F<vt+"
    "G>|EnIlB+>BA$}(Mm\"HA9FhqfukIb~UALvaA>%cJj~U_nuf@aL!~(XDAk_|r:>]F_|sW2i7Ahq,M>V=pDEFJ}8Lv#C/5,_G`<vF>CBk_pS}gNR7s(Xk=$~Y4"
    "1+(A;v3$|s#~KnmWY4+=qCiS+>oB8vR&7A)|_~T4HB<~IAZFl_P+kB7FiqIY~~(_C\"wN$_T|q}@JUKJ4sZ4}J_}y:C[BW:\"F0W8M6]a19uB\"ZK^[B,5K2E:I"
    "9u|46L/CcuJNTLN2(8#}n\"W}luCB<mr,K>5G6yFUkYX~lqZt45dQG`HAo_1hDx8R,VtGGm(Tn`C|}XorGBU7\"lj@fSFqv(tBRF8ApBpJ^{uV`~^QRt;,HMkA"
    "#(AMG\"6aJh$_4F_)q?p_:CCtLA+>lZ}~&Bhq,rr8^Ujq;yN&R?8vjHYs`L/vYV$A~CX+=>hAAAC\"+(kBFB%qFZRJ8A#rWLr?U|AtJhi\"HA5FSq_s8AQzt/+m"
    "bF0qm(7rQQ41>}j(~ETx#.L}V][xlRgDNhg/TB`~*_OJVBPXgF(t72M@hAMW^vVW5FWB_~wLXIdcGaR?qyrvj4o_2[kBiunG7yD/gBlP=z)X{&|!Kia&%i6n"
    "FoC&yKhBMi#zEAI?2r`Vh~6C4(q?M/1krFGJ,/Q_=)m6\"}9vW+q?#A$r/Vt/9v3Tmu]L&|0W:>eFQzc+AA4MsufL6G+kQAS:hq%awM~FRwJt/V)_b11uTLeG"
    "!s9,_V$\"^,Y=9G2I):]jg\"7v_NA_h||$y3D_${w$`Wi\"&CEAkntuxW=Py\":r6GT|8v4A+y>`U@1QrysEeXOQP2ctIMiG[touQ)s?JC?rRC+_z[wP1t|EJltr"
    "_J=VcvrC$ACtdxFhE_]K^v$hXFm_e@3Y/`;v~XCOG_7F0Z4Y4K?|5$&}dA^X7{_~,1xCX?V:&kG{y(rQG}kE~~ngW0H)5Mn`\"sp$WaYeh!z|k1_~Lo;BOtCe"
    "knj1mB#FTturYJY!>qY`/LAeiAPv9FULy}>X]QIL6,FV&_?3<)UQ[Fxw#;AAP<zvUN#~`3],9h9^WIAtWO;?%vAVOVWAjx:2/>(q1uE5xW)6wa\"iYKz_|$_J"
    "j`=~Z1W|b\"vyFKhRK?pq6>YFb~jZas;?7FC*AA$~XuR2TWts}CRE;/$~Kx}g>:W4Esk=^KeFfx:M4)FUhYWC]W+xW%Y)zQy|ZY4Mca]6F~AYP/PD%y\"XCA_s"
    "_~bGCAhBL?6C!r|L3FQDJVxAQA2WbFy|1W^L:KQ|AEg~[W!s$oV~4FhnAtd=l_<~AtQ)M?S|AA$\"@CKXB`~#*(i3@FEB*MnU\"}kekB5:2#0yBYW|)`!f=Gii"
    "Tez{@IMJ}Qrs>WY4EBsvynj(ub#~dxi(VKJFn(DM{KH}[C=h]EAA|s\"=X{Lu{Lk\"*{#<~EAzm+fL,/Pq=)av3KjqA*7}F\"(C.M).ZFv(2(fA]XR~0L@9cx&X"
    "ENps@$}NM_\"x@lb?L{qC$daDhB0n(ax??K/s+sZV6EUn!rEVlV1B&#X)34%v:v&(q?byMZEL|FpyHAeA<sEABA+>jA1W:>^K;vIAnB7vdZJBgAAA]KBtHAgA"
    "4}q?\"F6Cv(^)fAAA&_j|1WlBe\"#rv(6F<~9Wr?l_ezHA0EXAY4$^z_/C+>}F:vQV?L5F8vA*DAi|AA)_7vT|tW)_yqlB4}!^jnAAbF<vxFQjEA#~,YRDh$X+"
    "n4xK9yK&n34XKl5.UVG\"X=e]s`qydBfX>K9Cq#FhP?<yF{4*~KCz(8oZHBM?VtQVjLxq,o)*5QPt8W%>3Q7F),%>BG3y3SRE*=AtJVZ%%A}`^L,brth&W2>/"
    "5yTci6mz+0]vJ*GBi_e})AN\"bL\"]=vI\"$\"{%y(.`Vn|bqLFB9~lBa+cL(_FcfLW_$n\",=&cFYvJ*u>D\"lr{2k_n`&|9VoW<p+BPLE\"^XpBcARV4A8s4(m?hA"
    "W+DMVLg\"[~eF2[Ytv(o`YI_U9Vh~AAbLw\"lBdBdFOz4}1WZL/C7a$Ai|_)aLi\"VZXLAA2W8AQAEAgtRVn?^K:v_~WL]Kgq_)DAQtAA~~QtkBAAiqmWhB3LAD"
    "IAbF1Bv(DAMt1WXL}F<vPj)AAD9%8MiAPV^LP/ADRVZ=(_>v+([>BGk|W+bLeG(|$}4A?~&X>>TQl_AYTX`FOz1u|LEA<)+7n_(Q6a>2HM)3jyLCk]_\"ROqW"
    "Nv,(9>)fPDhILv7g%m)uiX;,7<&a^|o~sy]|V&K(Rw~Cs(4;J1H^Dvl_+F\"*gt5KbidBJAh\"FtM{LyAV}gF?kL4(<@7QRtgV/V8\"Tq=>!~,C?b6K~~CqnWR>"
    "?>byv}1K7F7CW+]6In?~3}DHb(SqUch+zE23fA84m!1`FC{Q|hq$rI]&%qdxnU)cRJsy#ME_KyK\"f^EL>r`@1Yumlu5**_^A5NsG[rr,2}R_aF9E+t_Qxtz,"
    "E*EBPA%V:_t~Z&h~F\"3(3?xQ|v;)56Zp{wgd@~fAZ&%>>([n()!HV9={2(<Vd^mI<s}~BAeW0AOw_sQ@FAQt{LuQitaq=``EL1Df]>cF\"sr,@%HAZzGFlH>K"
    "I.mjcM)K7ayi1XspS&O|:LX]dZj%>dkIJ&S@1)}FCq}~CH^|[,)h=(TtnWfL:>=~OZ\"voBHCtZmi~)^hr,,48F#~?rf1A=R\"J@Kb1yUE43).${Nx1KqW6~8T"
    "O&aBbvy|T@P)t~kB{e7^S\"(An|v}%>mHlLlZ=~ZE;sc_%&)~)_b/.hgFnvsEW]:>};!$$ANGdW5tTLby?$rMuQV|#>li}L{E<%r?!F%q4a\"]<?n>iEMs\"G>s"
    "}beiz/]N;]~e+K^q(XdLmKRn;)wI2KKDr#7M!4a{.!stB~a]O+Itu{Aw_:+I>.x|uZ2D,b`vH^QV6_n1b#lU?/{[X*DCM@V4&srYF/X}UZ^q3QI`x}qK<@I["
    "w1kN>:*n>r+:2FnnJ\".J)h()yu_(7v4$}~(\"DfuiJ?Jo`%RJsK<^;{^Im_5s|G>>7\"^vB*P_N`<(fj9LjR>W+O(NryL\"0WVq`)&O,bCtHYeLHB4I`sBMdFgG"
    "b&+>eRcI`s|)*G&n)vmLUWXUV095uXH4;C=V:Q^[aV*V8A`s||K?#~+rg@q?6vaqRJhB84h.^a`LT|0ZOmXXOsC#yr69YLPtv$$/smAb]JF~iwK_cN.>wF*_"
    "_~k^lt\"9k~OLdfOby?KV)|yX<@BAa&g~xQ/C(CG>8L(|!rTLo_;v<)A57GitkZ1*aF8vlW^L.>6CJt^LH`\"v$}#A9sn(2Kk_5C&CEAPG7H!WgGPqry&i;HSA"
    "[VJ/hq%9N&2L<p<Uv(1}<yjn3f&_V|QV#f+_Qt+}`|B?W|;)Isy:kO~v!(H_!~pqVJcG7v3({L$_Ct#}/>aFCttW8AT|!(aL_E4FBt1Kk\"GAt/5C9WiivW0|"
    "A\"IVXLv_crHgZF3yWBT@p\"HY$A}h::c4_`QujxqW7LftsBt%=QmCjd1*M?f|)BzibM/1jEr(\"F(k]s`Nm>J4c13(ZKT76de+pQ*Krah*!Fl_K&euVF4yry4M"
    ",?>97C]>5G3Cw}Lj3F:v&CX2#Z0nF/]j\"Azno}RJ&\"h}I@zQ61*sZ~xQ2[qy/C6}j|<)Bt(_lqEpPjHhl\"U~4~h\"ejuQ=C#$eXRF`_a*:qZL@Ah6A~Pw`eXP"
    "J?GI<XzgDn(t5Ce+E#$ZM~_~(^(k{W8*g`rAr(jBgq%aS&<?<xmY3,GV0n$}<VXFjt1+NC(~DBA\"ajIBxt>$$A0_mWhsE\"GDFhN?CqZqxdC_Kv0XrI2Mo~U+"
    "k=%_If5oY)c~~yzvujs,PnOc~|}S\"LeE3u7LK49Z$YCGaLYDWC_W51^X%**\">r&CB5sy])\"?G`v0OZgi]KOvo$BuSXknRVDXRJK4ur+>7K:}L|RtQ`LvlBq6"
    "3:,I!ruWm_?~@$`etVW|Ex4*<VF4V+EtK/Sr@$.s}K([kZoUr/dv[$xWnP/eP(f?,/AqqFb({_q]4r@s,8r~tZuuj~y3I|>>$L(q%@wp(7a1h(:>Fc&HC&o@"
    "b)*_g91%[:WnUf$Ag\"OXdFzt(_*eTLY1t##3U*UDCcBa<:b1?F?0ihcv:`Z]`K.CeuCO9~?p]XyiCAvWXXdF7vC&v?n_fA=JcGAAFVbF~y&,<Jv?\"F*Z|s]K"
    "Wyaq=tJH3}YVk}4F\"peW1KA_gqB\"Cf\"}ODAtmW1F(K*W>>}FXve^!4eY4RAbV],ADU=JR?u~6v/>cL!wKV$ATtBtOCE\"gVnLI@p1~$5WVLPD3(eL5Fiqdx{|"
    "*&/I{UGC.Q|FC*)[qB._S\"</9vI*piuQ9/>[XV|=HbCSoPI{|Os/NCH`Ly^)$}CB?wLETCMG=~K&qrgGsvh*,Ai{^,1*~!\"17)9&GB4`+)4gj?EB5$k4g_Gl"
    "zC[&YL&kB*$JnB.IRF,L9FaLF{SEbFN<>$U@g^EkOEYL2K;Cw(N>e^$TV&m4URq?%}aX,Vm_iVsB|LCtA*0Y8mgtCV3(tKF\"{uB~K?Rqg~j~14nI)<o@>FS&"
    "oI<_ewGYEM8AQWsUiGg$XY$M1F\"vcZn|SR;v2(JhuK#tr(ghCGPta#~l}4X74$in0W?FBYvDwWBAp[hMx$h&01}FU|`%t6?QR\">>>?L<YDlNk\"ntOj8_Uq%C"
    "hN=(Fq$}wBSQ5F`~WX%~H1u(cBm_0n;PhZ8QH}BDs?_V6I)kQX*Bm3]v>>#AuB:>lAuu;XqK).#$5B1F31E/#}2L+v;X4}wL&_AYR2VK4voVbLj`l_7,vW>K"
    "UqRtkB$_cwh&]XO@X4#}u?,&(|xVG&eLP\"b)gF.1sX9h9^~1[a]V8G81<XuK`R][?}k=)VKyg&{||Eez8Wn(dEVr%$[>`Q1[r[2<EB20,w%WrbCAZNbbgz]}"
    "eLo_2<kB~B$o^:TqiK<>p7bEee4GdwJAOcm_Z#I~hK0|xFmW>WG4=~B\"o~#^Y*3BDF5ypQV9u(V|u(H((G=~YVyKA\"RtW+/>\"F8vv(1Km_kq3rAMg\"3(1KCA"
    ";X})#\"=stWZ~6Ca*vX2K>~QVQ)`EBtkBuKAAAAe\"X*gBH?Tq`UkUC\"AtyW@}N/j:>mMCcLRzYh3LLrFxHM2Qz_u(B\"{F<v_shB_Khq,ThBeG|y$$n(*_U|K|"
    "{L.>IIRARXH%PZB*%_X2&y<AAzw(rg5Fvs*)Z~LQw{UZZNRdzt>}bXp?y67)B\"fGL4ia/VYGLyHwer>QtG4F,20W;p^X~]W/Zop#1mTWG!_moZ?VCGFU[>a^"
    "tD$Q95fM(t])m(K.pCL#Qs@QS_{s;TFP@td<GrM@C\"<h(=,>)h*}w?8v)Xk=F\"%(&fxRVnG!m41L2C8y@Va\"=v6r!~vtotu3\"gqybPf6o?MDrEw?(>1I(,FK"
    "&>Z10+~N\"]zAogg\"&C:>qLRA:>{RU7v}X4L?XLeuZBgAvrY4sVL4!rz3s`bv[oH7,_AwcB=)3KxGVxUVW)/C]*R+C^9CcuFOaE*_$aUJTRrv?DTXyGeZ6)(A"
    "YFcE(3Enit]``>M`:C.`73[K*H6#gBL`hqI*aX)AJt|X7FODUxB*p_6F#(%hJ`c1k+AMn`\"CeB/VgA(X>g_=|F7,(M9~UqP{c~[QrC>r&K#BQt]v5KdF7F5v"
    "2WS)i|*Wz3zQgtt+C,B\"K4z,p60/6FfrtiRLg_o}~>d\"JtQVnH2UaS2i<>={&vH7,._nR*iPNPsAgsj>nbb|iXqG.ry$KC/=9Gaq0VUQi%sx0I^(x\"hB*b!s"
    "Vn%JIiFi$TZCn/gQ<C^L$UdI4$b)=)Etd|QMX#`xJt(>VEy6i_b|i\"CHCq[RWnjS\"uhAxaa>1)jqRy?}~Abv(SiW%^;vMRRO^RL1+(15*A,r$APw/`@V(~i|"
    "gVh~=><CQ*5K>Q4y$UlB\"M?~;)AAb19W0ME_:vbZI)p_2yurv(/>Z1|$XLDBavQA,?zqGWYgtiroDf#K>K_|#}f)6^OtV^9>/.dv+T)Ayk^X#AYIWuJ>4F/C"
    "er#f&_9~()[>M?qyBqNVO3ek`;^jLjQDQD?CMWC<6yIJgGKCr:^UcM/CuWcg]_gG;Ci+KBqF6oxI=Po73}XLp?[{=$R>%_n1OugND?ADL&#}.h41]CwIN?W["
    "VZ~~5}orNxi)KHsfwTxBZRS\"h1t.ok2BIAQA/<o~57/XXX;b;C~auKJ?YI],/5h),k+)V&5TgtVS$YH/\"C>u*(T@FDc++rscq?[.Dac\"iqGC.A/r8TQQx7Yg"
    "8dpd^|8n#7n_qy/bq~*G%_%v_@,\"*%.@<bfw}o+>.BZfv(,A!{[a[>QBBtS14tSK/[dB~JC_yt(sg4<i)UV&]|^Kv|BAgGn_rW_sx`pF3Tj6LQ8C$$xB<K<y"
    "0oB>@J8/qyY~ALCq0B#7UL.[HYoi.?SNgV[VG_9~fIFMm\"7s6K/?0x&XVVeAzvZNn/rs(a=&|E>{i#9hn?>~YVW+wXODgDhl#^/v7z9&7}m_3Fh%i?.ymr>&"
    "1E1nGx3UG.2nQ4&C{EeCD&w5yR_E4$OTS)W1Vx#<7_O|j|I~!~]kkxMVuEbcY|!$kb!CCt%q4A3un?tKSAp=UXm[]$XXYG4ve+q?ZM#t1uX)\"=^[uyosbL2n"
    "Z}SLgG!vSV~~|4qyYtb)TK=vv}IV_Kr1@Q7TyQ8v4}B\"%_rCk##X(FO<Ow&;o=R,H&hV*/5[[C|LKcjn;ytKTF^_8B4}^QbD|DFtMB|xKEUvy:]\"9Vp?<Cm+"
    "xiw`al^a=V@KxnRVqW)_;CZ=9kSk<m$r_)l__EpCY@6LZ12ulBk\"^X$ARAIA6CK&r?BAAAm_pFAA9~hq*T^j,hXUqS&C%F?mw}.VV~MCv(g~l_/C=~@JAGgt"
    "AA+_AAAA;vpqAAODvW+>hGOD=s$AH7OD)MwQ[NuW})]Rkt0C%hiAItBW0K$n)s1[~}qF.94t;Qoho(=2e^Nwj,8~D_=s?$+v(\"%yuu)^cD<vAA$3))e6WS,s"
    "zv(A(_fwDY\"MiNe+)ht//C\"ChU|*uDJt$MbN%q{(bN&MNwh|Qlq_Bw!(q?8Lhz8n2Wj^vsIw}~@/j|djVC~^qyzyY~{L&k$}Mh2EUfr#?)v_*k7vuD9L5vDc"
    ",X<QR\"oB$_[~`~YZ9~RqQttNq>JJi&;}mBbv`D4IH`LCeu{@cLUIR*#Yy!qCHY>L+&sy_a<@5E*Fb|/&2MMyB&9r3*mkU0AMp`TDRV_fC\"gxbLyQeDr:^jV@"
    ">yynR)#G>~VBatqW[{,buW&_;vRV/5mVFl\"`q3yQwqn[_B*^`L=UIAjn=)?X8}1exoi6<QTwsZ95|KAtY{es}EC\"fsaMssi|<)aF3LB&zUpdjAjL1L&3eA`F"
    "5v*sqK{G_h&CD7Id,h)sf|3S;~y)7rEO@tBtSCCAb#N~S/htqB/Vd~$|1W8A3y_s1MobWI~*k=Q?|n&C?jqWfw!rM)EHCt[y1p$_*_%C8M%~yn,(1i:`bMaa"
    "/h6\"xaTL(~Dw9BI@&/ZLcU9kwK]HeZsNn^$wc+.5}~uD)EN>(\";CL]4L5FrXqWTRXiAA3FrGF!23}Srzo@V6GHNA9[sj*LhSN~>`bJAAAGLCF!^s{F+|bSkN"
    "!Q_4JS!K[J#9h|*KEM+3;)li2YKJZVh=]r/3BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAeAAA+>R\"&T&_fAAAAAXLk_"
    "BAAAQAXLi\"AAAAAA|FAAAABtIA|FBtT|^XuceAQg/bhn7CVNrAWU^&VQV_f(5*[E?A6V0SiUjF+HBLvF4b[=,_Rwn^l85}WLdB;j_@fAXj2)aD@GGO6Mhqsv"
    "{Lk_21>Y1WZLT`%n{X!\"Zy5KcM[Hz,ns]EX42Wu?#\"E#a,.?/Ctu#r;R/hJYoB@QGq$v_&1F:[QVE}oVd1PZrr0hgqM/6ICGj|,Q;LS@ED~*M)DA?$wNo>4?"
    "iF+:)V!vc+bsaE}r<a+v#}roln[~aS$kCVw}ZM^|5}]WgR_e|}9h7~{7.rq(H=*1jxDM%FoyO*M</E&q=s]jsE$q&&@@q/SwYtSLk\"*W*h&_<CvWSC|FT\"<t"
    "n`;CIYPX7FaIl/kAiz2}cB5FJIa|?4`Kxnr_7vAFpy?Tn?I?uvTZ2(8~C\"1~ULZF!W+>*_ry1WEAj|]CTXAGAz2u|Lo_;C2WbL~Fa1T%o5`EfwMZ*R~F8s]v"
    "yiRL:vnuUJ=b#nCt?MUMKFSt6C}E}1kBqKJb|xs_rwz*jLiSxWA\"LGurn?DG3vUZn?=E4]6vL@s?0|/a%*EF.+uacs+bk_cu`JxLwsTVgDrJ3L5V8M]EMD6n"
    "gB3,fp~C=t/>.[_X2K\"Ab]:XVjB~cCft}J._([PE4kTK\"C$rh[^K(|{T(2/>Dzu+QV4}%qB*v(i^gt.$^)aLrwb#:VCGfAAM_K>~hq.h!AEUBhF`~ClB~~n>"
    "/vZ&<)zKRDmu|L&_Z1*(>j?Jqy<)!f7F;vGm{sXIMM;,U~0~,8JtfsZM^>cBlB8\"4F]>4FLD^XuWg_=pB*@V5~6FYt:G)U(+hxM&t>Wn&a(fgGdw{e1B@Kz_"
    "B\"WL9F)[1Wv(B\"__XuAM,?9v^vIhbLeD=)0*#G\"C4}t3d\">Gr?\"R%w>xWmBFsy)).*H\"7)nsVLo1\"F8,Z:\">jO*lg\"]d3</c^{1ZD7tWLf~,%K!\"BcY@kbLF"
    "#+#pLVV|`(v*iBoR5CN`F_GRIAkMfzlxDLvLvqUZHc\"}zk`rn|/&*I6F$MBZ@Kr,tuT{9D_]&<}F2[6C6r&m+_KY.sURlk+~ZC$<.CLE/`*>+h^X/V4:=yxC"
    "[1sPM1])?)t@oFR*TB/LpI)X9*Y/(>tB[JL?H}Q*B\"xQy_oViW=P.Cgur?GVeqM|cZ@)A|,$,vQ/7v1u\"LgM(\"W).Bn_Z})htLYfYrC|BG8yPDi~zK^|e!Wu"
    "kAXV2yi_/y^v7Iv(Z41re2LW45zEO&:?51{{4Ws_l|hF?LP/jtF/JOvL{xgtM4XMvtiaYV.>D(lW[qf^Bq]t7<^F_xmrskI`}s^nj(;W:v+)B3!OXLxaT;{9"
    "wDt#XXP/Ct3d7MSWHGh|_4A5|CcxM@sV{k+$q?4LK11u@Z.>G4]&IQ|~Tn4(R&#Gry.&&KgM@{nY`J%~dZcT@%|G`nGO$<u,RDVZ+>|ERtzvqKb~aI6auW|~"
    ">vtWdN+>jqUZkL9FyAv?Y~W_;).VN`y|t+o=W~WtI&V&I?A$D/r?tW>|W1aL8E8pBa`4..]qnBEMTLPq<WW7>P`H:oglJ`^$`58hK:H<t+w[(Qmq[W7M\"}Pt"
    "tZ)VI\"=edB3Fi|2Wt[hG2|R&}J$\"_~]XA_3_W+wN=(tyluIh3`H_bBnU+_YL2WiWbFZFoVr?$~<vq#XLv?J1sByK|FhqIAiAuWAATnnT4AAA~JyQBwvB@jln"
    ":4kEEAiAAABAAABtAAAAAAAAAAgAdBAAAA$A7vAACAuWyK)_Qt_sQ@!~wqCtCTe%h;g|LA}y9WhB0El|PAL?Az:v@Vk_%qS&EA.InW.A~FC.Zu)_u~/=}JcG"
    "aJ6,X2`Qt(q#GrJ`B5Gu8MqW_|ODeeHh9|hV512R~Cr&%CH^`HSR.AyqGxy?_(/v#_M@o^$nBD6f|L5yn*;)(`7<>smSz/5IEq\"GV??[&r`e\"K7A.j?KgD%q"
    "t[FAlB=)hG.Cq}cJZ;Z4a|!7}FsF],{LpResP(j8#~<v=X||*Mfw&BAA\"sC\"]L2k_~COq>BW/$Y)C5uQW+y{w/:J~*~Sy)~FqSyWCA<Z*j8}#p>)OXk_FL$$"
    ":>BGSqmud~d}RwB*4ACtsBfL$_^|KEXLvRqItB.Agt!WXLE\"aq^A~CsZZq1FAtmB3r+>QD2WFVU:4Ckn8A<stWTLH`6CHAA\"AA=J)_qyR&2KL?QtdB|L)_6F"
    ">~!>CLFq9uiL0\"o([V+>w|`~}4P@<sVB?LZFD\"K>0LSq8uTLN/1q,$2K_K`E6CHXG\"SS9*BAwaCC,`YItWyK:><s*au?ypn\"r$$3R/<`QVEC^3r_HM3F!C&d"
    "7AQdGD;X_}U}:C@L}}JF&`SXy{K7!(BW+>M2;sRCk\";%M@:?*_kB)AAD>WP@nOdfE/(4B\"aFEZtY{K*nM#Gf7bB|]&u?0P1L],cJbF9$Q&JVMQ!CLU9>/_3F"
    "wF,v<KD\"HXdzQA|LC\"QVAAT\"|L~~<~sB|L_K!~^XIAS|W+5Kr?by;,*>^KADR&_~3_y|(v`V|FPw;X+>YLQt0Bn?}Fwt2r:>`KBtlB`J]K4ys~kBEASV6WPX"
    "qFnu/CbF%n>r~~B\"*s/C+>#AAA)_;CJV2K5FiqmW2K)_x|&Cf#OP?~A\"TLNhWrE*O3wVXL;=@kyK/y/`IM;PdAc|(AFpM$yW>~fZTXS@<slZEASq.}+>aF>v"
    "u+/CJ/#pK\"OHPG^aMs\"M1ki#TkbRtDtEoU!G5ytudNa_#sHb+q`FrC{sIMuW.}qyy=j_0kq}<hcGf^6aoYkGa4R}Y)Kbc?UB;HH/pR%$`&f\")|YEdFo>Rw,$"
    "^EhA|;,>tIx:PE13~KHYd4=E&9q\"l>:C~}!WA5IPsW@J6A`)_VuQ=vBtF>aLVq1Wr?BGnFcB^XG\"QVAA~CsBl=FAB\"\"LR/=~/CwMgA2Wn?p_8v/CIA*hd[N>"
    "K?1_+$#ABtHAE_*k!WzM9F9s]C2{$~wq:XyKAGtZr,d2iA|DJh5FnFPY%K!A4$]&4Fj|Vu]X&\"GDIAS|}93(]|gzm+<AQwADbXrB<sj|AM$_qyY&LXC\"#`B\""
    "!~C|%C*h5Lm[v(Kj+.:veTf?|!y0JqJh@K+>bSX){LNyOA_Kf|NZR~,G]nB_C]qPDlpqS614E_|C\"2j_Zl}`;}M(/h{.sBi>\"cWYT4j_rUiS~+4X_[2uaX*,"
    "Uq1n.<<Pf(nasF1~|vOuyW.c`KXZ:jgGbDGcg~1KbJ}T\"uq?rq}(T);>54or,A^_rEI5l_!~rBDKXRLy`T%Att:o[~]`pF;)Zl{K2~cZ`~{XKCM!=&a>&|+u"
    "~Jn?AwI+6(</>sSAB#]HyFM*AL\"FR&]=?Q}CK&[>DAd+?4}FRt6C=V>KH4PY?t=EPwU<uWB?Zydxl5c^0_R\"6FpCdB*V_EOD<)#}y/$|B*P@n_ZF@$F:S5@n"
    "sW8)@(yTqqH37Nz|)v\"X_KwnWu.V$\"uW+>bGSnT|P)q?[^tu<VXFhqtBU)EB;~R&yKoQR\"[>2#r~pdSBHBqyrC|2y:Z57s33%_,xjy^v8FI`}r[V_QXFV+85"
    "AFtzG.<@`F7D]C}@WRaf~Xa2O@?{Z*<A:v+eO&8~xt|bXud~/~xF1*kn>qrBd~(_~Ih|_4sRjwr#1W$AhA~(/ypSP|aFk_~)/C$\"dZcUU:.Ij|W6U!|>?`X4"
    "Z5RqBVJVuA8UFhU{3yG+|Lj?DwQSqB3R.s{~kB\">DtCUxBW!Y19ZB*9\"xCr{[:$mn=Y)`K\"{HD=NN`}~/Y:C3Re{5C;f|RA\"}oH2BGrClBFtKWdCa%b(;W/y"
    "]Xu[J@nH1uc)~&2vU#OX=9Bqw$KLzL;s$$:jo>wqo(^),>6FZ.=JO@y_A}!rk?e~kuw=dA$Fv}I?IioToU<KW_#}&&7K(|1B:&&/r1J\"E?Z1qB^X9`Uw9~xU"
    "E_uqfA5R3lf+OXe0:sY*$hCG_3Xu~ZVRqC=u|@N>Gqmuc)vQeys_T@i_Uw\"C1*#GpyUq]hQWrvg(ks1/31{G`BGoxk_v5iUKsC[}M)p>SBZ*KY6@2nvT|M(F"
    "|_~P,CjBU_qq}~UR]E:v,A4y\")1K3FnL?$Wel2\"g8(u?{FQAn?hGeA0MZFU|QVTX5FBw&v7}zK6FBA3Fgt#}A[}FW|1W8}1Q\"v&a6a_G_EiE6C{EfGqleB6}"
    "oF#W&T:U+[rcp=1&sFIYY)Dbl|>p(3W@F[<{sUuE_3d/.>IB*k>[~XwCGubS0(M?yq+B85o?IqKdY7{oP?bn)J#~HCIV$AKFB\"<~$~2C$Yw[[KAwdug~WFgq"
    "ODoAn;nyopr_tp+)h%xKev<~aDg>^[jB?).`R|:CRV>cW]O0.59EI`aSuKbFhqP(M@!~3FMEp=p_bGZ3SC._hn^X$8uW|[^Xc]|QU\"uWBB5FTqEYpVDt2uPj"
    "jbbyvZoNLB1h@TqWC_MGIYx=O/iqDUmicH3Fl+jX)>VLduDMPFN1>o~=W;wKW+N>k=wNuB0MyQvwod@jm_9p6C)MK@:rR^>7?X,Q\"a<M\"S(|_BOCR?CD,r[i"
    "\"<m4&,HA`_2W:>fGsC%C9h,V=m!Tgg_F]N5`f)&FOc*XS2cGEL@`*&x!7sEck@{,AwvBML@)Dqgu.@A~<p4}Ssp(%kPYsAUD.:Y=^)[jLH>>%<u\"gB<&Pt:v"
    "^Lp?MzT/_J0Ky|jEju,_0n,G`&TE0FT|L@3E|CD{B&n.iHkWos6GcvSqn?hGeA4MAA:v\"ez[aFPDh~S/ry&X.5u/Yi5$B>{~T_MZSLg_H_/C>2n&w[0O9@bK"
    "zE<{5>N!ap?`Rqm(%_4$W|B\"!~0WbXEA}wW6EGi6lxz?R:hwCsh~/>V_8B)f@Qhtj#OLdL(nk#]L2KSqp&^L6s<v0B)A=seuIAAD_XqWk_;vuWEM:>ht%&_Y"
    "EM*t.`]@H{vw9WEAAA=VA\"B\"GDz(xQ6F<vd~YF7vB*eL(`\"pXurgjnUwX+Ah2!h\"6Ku:LCu+UNZXK1}w5KF`Bt.}oNl`6v0v?XoW8vRdY)cLLy[o2(eFdD#("
    ",ADAAAAAAABAAAQtdBbL5FfAAAAA^Lk_TqIAAA3r$AfAyKAA&CXLAAuWAAAAAABtAAAAAAAAtB/V8s)~c8DAAAAABtHAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAk_AAbLAAAAAAAAAAAAC\"2(DACq^shB5FT|sByWk_htlB=V7FRtsBhBk_7v1WG>8Fj|{T$A6F(,u+<_ssK*jkC,8yiZ:hL>qGjEP2}Foya12W7L}I"
    "k&cs/N|yH{5KBN54Z=)hhh`hPYOXeSX7u(vUQQgGmB$3$G9?Z(y(x)@{.$]X~~+FZqk~%!Uq=~@@~Ejqe(]@o`r~UEhZfVQsP03?!btwFxT@xQ`kH*xK_E(|"
    "kBGK%~/v&W/5>/h9ejr?jg#~yn**?KPGw(uWr(T|}QeigmYCRA3LZvBp+&o>6F_X`>/.Aw0+zk/({OI^?)QLpik|Wqv(kL~Q.ht?iniyIYL/{_%Xs};>.IR\""
    ".Q}1woAty)`[h|e?=LOzD&F>.bAtHA5FU|aq|Xp`wtVBhBrVi|uuz?!F3[hA~Lbl^ynIk_aI$}?A/C=)=>1Qz,\"vRV<V/FD|NO(/m|1W}B9L1yRb^s`FiA,X"
    "lBUtiEXLWKiq&CjW}ECtzCvgg\"1WAA51Lm)hZL;s@;+MNL/e=#UV&_Ed}bb?$LG`~w`q@Q7C[`uWLBXkC&I*pQ=nnWZqe0J1Iq{+KVynT&wM|FR\"0M1Fo1Rn"
    "mU~}N2(%U~^!9s5C$MzLYFeBQ@tKLJAA>KQDvWQ@B\"?pXAp?Rt5}Cf?R@^s+{|V>{C)D$MF~fq\"Nae9HloJY6r^W?6x_JhV9N{M^JCy_1|sBpuyR0_4$mKIc"
    "NyM_O7I?uwvs?EA_{xgV$A/FKVlUn?;C))DM+_>vuWM@&~_[1(FOVFk\"Z~hA9~\"L~Fft%a8A=vIAg\"VZDA$QU&8wFTU!@BsMnDaCBt&fj,/v5CRB&_U|!r|)"
    "]/ZF)).V_Ewq\"XRh=Jc1;X%h/Kq~,lX)6=cp`~%&,_Bw0B]|I`OA)<8Fsvhq8Axqeu=>B\"BAIAA\")X4MaGy|dBe67EXLEc4kHN3RT[}{P}Nz9Wx[D_h\"AA34"
    "!r73`L]qgdSL6y4FuT]23]Zy(swtI?pvC|<JbFkq%X:vgHpFK\"*AxF.J;(l[RdCOAAE+:fnB^hLy2KaG0FTVhiP[<su(%hLdxnr#&r9GPMOZuD(^hBr&kgAm"
    "sD=}TL5Fxtj&iP%=GC)vZsyQIIY1/*s`=1KRym}RQz#b9hg>EtJY`BEGAGaq;jtL^6bE(jUQ_kxyqWxFz\"H}vLI`.$UhgGZI.}P@BG=pyZ{JO_+3$$K(M`wt"
    "zBfLeGyq/r]Lm\"5al~}L7C(vRJwE61K|u?ERvwqdXLiG%qB*h6I`#~bn:2+~aveWXLC\"*rwMVLgAj?7]R|PV(LGWj|$)Fho>V`\"X#M<QL]sBoUl`;CXAcAKE"
    "(rXL2q.(ZN|R:L*uxN(FvAMrAG6FuB%hFBbFdZFh@Q{6yC)tsLcq@TMU3F;sOBg~dA}QQ<>L#~cZ=2{Pv|GxiKoA!TH&H_m|:UjP\"E/~GYhB/|JCbSk%65A|"
    "6PV~9\"X}?@<_yq/Q+JP(6_x}piPL4[=(Mh_EAtK&*JaK|rzC{L)_it;)iufT_Qk#?4P)9C~:&&4FYsbcI7SiW|49w}M?ht)X}~*_6y;FyKY:3C/`Qh>WkL=~"
    "3M\"}aOnp<@G`o15d4ACtAt}U/V4IrS}s_ERqCAJBvAyKCAdBEAAAAAAAAABtAAAAAAZFAAEA7CAAbFBtAA3FSqAA5FM1]bsA:CIAC\"HAAAv(DAAAyKAA_s8A"
    "6CuWDAAA2KC\"3(kB9~RtsBv?tWOATLX~Pttue?u/znoq#T`38CdB=VAAAAeAAAAAuWAAAA)A;vJVfLm_BA=V~F<sO+1K=(3[SqRJX~AAd)2(<vlu>>H_3Luu"
    "jL<>gA7f$a@q>~1LgAZ}^U7~Sqbq^}rQU\">7,>St3CLjK_qC#}s=%>dt:vRl~L16JA0Q0|PV+W%^!{L,}sCAV|B>;>kn$a^|}FH`Ry:rCG1_/at~*_@KpnH@"
    "EB6C@}>Ll?(|1+!>JCEqzS9hB?X`=)+AlI$(F*0/g2vT65](BAY)yKZw&,:X`F]Ko}hJAMOD>TT@\"F(tW+pW:&E%mW#isQYi>r&C]E`Hbxw}uPV|oqoZD~IL"
    ">~L@y`B7*e,T$_0h6dzfE_by8Cok<KBH8rDCQQdt`Xb)xQwq!u(4M`1_wSiByKsy%Cl=yR&q[w&3yKo~o*b4$~7vFRGCSKCtJVt[rQxqsZbsH`+h^v8Vj~%q"
    "_s5a8_qv@TyK|F9v_)<haMStB*5i.>Cw2W|L~FxAGC1L5{:`p?L>@~!>M&,?234r!?(}Hl~:7?|Qky:`r?pQ[qdBY)WL3]t){f[LZILV]B~*ZLVD.A@{%C{~"
    "r_DO}(?sTFvv`Olu^:CDb&@}yH?ySZ}4TQ1k<vUJ=PdD7k^X&\".oDjV!}Ctx^)<_(NySShI{+48B`)I?DqVWAMZ*SqO(ZN*`MJ:,OX1~(nK&y(cAznKCf~71"
    "H+iX!B=~y|6Wx?\"C22tik\"pd;;bGz_\"aFV3`<pdBqW|FsCSE:VrP9yCY1%8^wA/VC\"HYz?,>by/C4AfDjy=>K`SqRAzLay1W/~aB>d|brIvR}vnB/V5Fm_6C"
    "XXg\"2(DME_Bw`~kBf~;~ItAMzDRn<,($_c=QK*5Zbd;_PY`Bi&/v5ke(9>MvFH1~%~~v^pB>c~QqK|F>2Fs~9(5K~=cy^).4|KQte}_sq\"JY>&jW*[`}9PVF"
    ";F7v+z[)<^hqr?aRI`&,>|R:On5a>={J!$Kc{|2?$pfZK>V@=vo}(v..l[i*@tZSi|^)E}i_K?qnDA}COZvszFIlmZlifhOAPCVLoleYl=Pc5~#WsMeA[`HA"
    "8F\"V/VTL:vtBt=bGJC#o:>9A+BIAtDM/#}y99FIWY4WLQA|j+>*nc/45zLh<qyS>%_#sGx<@1EgD0s[J&sX]IW6W3(Z1~`%5]RYF;,Lo%~2_)WS&`:d2DFkU"
    "!_)wa,FrwMjkJ44M#?E6^tgUm.~CSf?vTK~MQtdNZL0ksWx[f:aFn([h]WRAD,VQUnv(?A*n?}M5tQ?|!Ws%!~bIuB9Ky!>p]vU~{[i|Wu0AdD~sNJ@EdAqW"
    "l?Sqw(;Aet3&.>{3z3!]c1m(g_={M2}^0;BS;gl>m;9T]|&,h|?$L4F`5L=sN&g^U40(!>5Ml|w$=`X)|EA.TgN;`B\"C1~<PxquBRN]K~IERr?VQNG%$8[N?"
    "hAU)~K%n[FhNS)QADXcAK&Zi+\"RAk]8sKAyK0_!Wa+VKoCr:]j,HN#<9oIpVzqW+XXs)t_.},4C~dG}o$YfBEAZ~./l_N^Et+/{0T\"N/Qq+r4<4Rww$}5iXE"
    "k|_)6r!>wvvS*Ahwfuv}\"~nLot#YJ?9A.)6L#tVB%`Y~wQ\"Xv?3A#~5>IBFDZV(Hp/ftHApbCq!Ww}(m9t0B0A0_O+Q@u?QtuowY/VBzv}YAo7YV[>r.u|~6"
    "tWf^Nq8v*V/B~4C*tu3RDtR*|X:WAz1WpNXLgtKVGCCA/Cr?j\"9W2Kk_:CdB[JeAVBM@aFeA8A/C4}DASqAAC\"W+(A8v3(=>7GSq:C/V&_Bt;Xxdf8QeY|r$"
    "=E+|M|Dv,?={.T?Lh\"j#>zL>BDSt#j2RAD5yiX7La1<)%W%F.CtB/VwW+[}Q7??Ke1_XnXsWkkZ#<4*&[{c%=L5F%ELEjP%~m_pq:&.Zh\")M=Pqv\"].@9G[_"
    "2WB<2L9yuI+L~~w1vT*hmoD|b:D}Eh$q}$_JWm6yA&Mh}Eq?YA=JfCeE7vY#iDa|Alf~Y1+GaX2SBtqy?VrLnn*rOvj?=~Kt0*5FvZOxKXy)31`wjsKco17#"
    "F50}Z2m!Y2^r%0.}Mt1Ket/*2#i>+1cx7A?t2T)L|(!9^aJpUR77sa\">+N\"y,r*>v;@p1(CK;bbwWx2(!R3I}T9tIGlL{Xl}M`up;P[6?Q8~qW[VQQcyEUFh"
    "h`bI[,qLS:m_xA?L31|}iLi_GI+W4MFh(n/$Y_cFuC1+cJeAwW<)$FPA$A.F)X:*dj,nv(SL%~lLg(CChHBt;)<[}FTqitE@1KLomu:>6Fhq(Xe6BGa1%nc)"
    "=Q6FUZWu*`,KK&!rp?l|7C+Cq_ftCt8k6e&zMHsk@Lny/*9h,?htdB3(bGAD2r8Ao4\"vRt&=]h@rc@:hL2M)AMqWXRw$OX:_cAe6.>8s/#dNnAxyj##~Ki<,"
    "]L2Q>s8u`=S|Aq9+5hfFPwAAg\"M/2f,b>C)s&>(};C}o;(#>wKhoU4cnbIfuBWg_U\"H^WF(kKq!rnhrFFxyKv`C|MZhus>SnVu]jPKzt;X3(&Fvq#rLf!c!z"
    "r#3A5C{~S@CH:Fuu.A7ytBEAPAn?%ANx9K5LT|#r$A<sJVbLN?jqtZ!>jAn(DA_^2$N>e.+FA*jUi\"IAZFT|muu?aFAAuK&_QAj?BAv(@J>QQtIVyK9~rC2W"
    "vLaM92{,v+AMY|],4A*[/C@hA\";~cW1AvwnuK7m`F7{){eWKn1Cq)<#AhSn(s//y1udN).2A5V_J^k`)~XFnKG6,S]wXO|L<HL:WODOm4g8_Eql!hB>QtGVZ"
    "KJY@NQ5a!O5L}FHw%>%mAw3Fw?[HZqU|&KFH0qCtgg=?j|egE}tWOM%Tk?,WhzTq{X~L7y0u2(cGr~IVZ4cGi6KclVR?jt:vNl7^LfnZ%>=`DoB*L)yEt1>("
    "4NuEoItZRV1Lon/C*h>L}C;$ttRXGk4d<Xj`onj,*K@ET9!}+h3G#{70S+UVmx%qm6@L{C:]j~G`F[2W:CjP}yGb.~P@vADO$GRzsX8A]|K+~~LKftT_[v}("
    ")nvrZNN?7y{~,M]Qw|trLvefX?NWa]q=CwkB.XeMX!|$5Ks:ln`%Y~`EA\"u(J>4F;sW+UJwKpFp:lKC&C_<p/;r)I/&%)2I/B[hq{{;.ZC^%}s:^4y&U!OF`"
    "7mI\"`]:^!N/q5>~^^%XU6~C|^~rs,>Z>KauWf`QtgImW&\"0Z40U/I]y#VV,`pC(C^@m`n[vFR&$~no>T0<UR)NmY+>R:?~RA$\")sXLIAL*z}OV?{LWI);cgJ"
    "\"X_@6F>p)UMApF2TFOvQcGrn:>&RN~_DNg=`F_>}`>|]@ymW@ALvE|9h]Q$|;)S;&FVI.$@Vm_]\"=JyG&|}CDT3;g_pTpBp`?^1uNlOR`E!uv(dMeGfuXX35"
    "0_&,hn^0+j2r=V(~xs{}XL[R+[fWP>{F<C{rz?:?yqw(]X^Ll|TEAMI\"4(|##.zn]yiX`R^A+>cRY|t+w8iY@$!WrD)_AtS|DA*hAY!KqW=vRV$AdD{shBCG"
    "YIbaNhf~VqSVVJD^vmS_]se\"Yqcs#^z|9r?vbRdzp:uDjJV$eZ3(8FfAEVf\"SEhUq:%qWD^X\"M:~{bp[YR(neuQVL`OCsZyKcGa<h\"AAqSJ)M`7~$CmrI/.Z"
    "<r|)mna!~FlsAAzqsYgAE~~O_W.q$}LX%%\"@&!])_~h\")AG_lZNVsWryHAx?3[]9w~!~i|TqoUtVbF4}w^2Eik)njLr`OwWu9WQWet6ax[6X5F5SO2s??~6y"
    "gBM_cyE*+jdVUq>)4=F>_O>DdN;JY`|$Fm?RA(,oqXaGl[RA!_%_\".m|(\"U|iqNV!tND1WFG2IaYR&|R}I[r0L}QfDi&3}9GV\"HjgmUIuBK[Sb0IUZY;eL/v"
    "r#s[GBOt)WRJ?JF\"mik_:Fg&V]Q?Z?n(!&xQ0n.CPLU4WF4}Y=(>G|\"Xt[TK5C8~v4h^(k=TYVl`x_SVWNpV`hHVEADty)82EBj_R\"#=>~TENV4}rv4:k%F?"
    "wt:X`VzFtF7Xp%k,qm8DuK^9(_vTJh=JZF#TnPjB6s@`4IiA>~6K}K}[dBQNdH;D0uuKI_U|Z|S>3<Q_{%f?nAD\"LWSAcNo\"@o~=B_:4Bw})G~[n?b*YpVo4"
    "f/c7||hkXj_N0K_|wX_N).:F1Bgkmc[@nG5h??ftGD5JLRODH~Tj#>oCTE/V8A=(?X.\"s#IVA\"3ys_X@R:cysBVJ5L=~^sRJ&_bygVY4`KpF,},APACOj~ZF"
    "_~5*V:@qIAL/iHMR7MCGW!W+Xge=TqNZbL|F<vV+]X3FBw/C=JAAJV4Mp/0|kBn?`EhqP}[zZMB\"S&xiFBRA.jgGktlELC)_CAfLbLStIA)_D\"uW&\"2W_)YX"
    "8v2WFh:>dzsBEA/y#(.fth|[9rA}Xeh5?I1*ZS.vPqZJVL8C<vS&O?mqn(sB8\"urz?5EsF&yqW~~z9/Td)U/BA%tJ?RA!O8}OA,A#{BYy?T{qyJ&[>j_4CdZ"
    "~=oBGUtB73;.1_m}wtJ:ADH.^U#M6L1+X)Q`Tn:9b)l.1xNZ.AKf~)v4YLtvJAKczqo}@h!>vn>e8M#L<vLEb)dSxn8vPAaIrVM)#mIrL&Z%6RitcE8<:>II"
    "jdkJPL;v]z9*v(u\"ai`Y~1r)/t+B>+|}q|N:&6icxNjBDqWZSV]KgqpS9[h\"AtCCc\"qHv?/_8CL&)J;(l7c+[VKi)63*2D~FeG1W73UQxTkBdZN`,l<){Ld~"
    "~0Sq}IH/QA4}I/liIYqK2K[~ndj6eL/I@rqK_WE\"t=R/czvW0MAMpLR.AY\"coL*,nXwW<vnZoY&G*6qYxiEARD/4l.z[0QDX]Od|9~Ui\"^T|<^0Kz=5p8{&r"
    "|K;spvHV(=;FWBuW5~U>,`pr3}S>H|v}\"^B\"B|+@}~SnLE0%WL1k&XBKES1w=E$Y//o1cxTLeGpFkZ^Lo_Qt^v%wrBny+G+>%GCDgW)V&_?CkE<Ajn$F33_F"
    "LF(vhBAGStPAqWlqyC]L6mSqQ*B\"PQ414F[=2LyF`RP^HK/IY&O2p?3I%SeX]K^{{}vgqQhw~s2(s^*Fg(?L8E[pZdsSj^61iq3MN`]|z,F>4FvqYt6Wc\"NZ"
    "[Jt/WIIY2>[KIITqV@EBG4Q(*mHGV|9BV@MX/]=)o=%_lL<~u(%GAARJEAuWdBdFAAiWN/jnIA+>i|[sdN]KNDtBUE9F__gYXX4KGz2BBt5FHLAwlB~FQA)A"
    "xqJA5F;v;v4[kNcfm+&?\"L9vlZ}gk(&6bxWH(F:lJ\"{`(_|z:;BM(umO8$?OE|v`M[BAx:Jjj^etsuavWLMi3r?E;/i_!rf40/MDW+8}z!T|QtoBA\"Bwcu<B"
    "TX(__)q?g~$nuuYld~XI[XXL}E:CM/X)xK#pVB1*I?d2>TmK9FWC6y!>^KNTx:f)}F)IY+2?7GV_cS4[^4!TQAVL~C?$y^\"LIyL&N`#S6Tc+Ftg~gt3$w~9E"
    "Cl1B2^HA@rv?9G?stZ9h)U`xWWqWU:Wr|$!(O?Rw0sS>f>x|]X<@R!l4YY`~}Ln7ZVZ~7GDK*(@VdFnomByKOWytkZ`VYMbF#}3Yk\"RV3?H_C\"+>aFOD<)T)"
    "%GBt!(X)BGz|/vk~kny_!r$A=s^Xl=!~\"CdByKbF/F]vDM3F:v+WqWO:3C9+[&/?;FggyWgGP2*u[>LG/FT\"[?2Iyy]>YLuD#([t<\"PZl~/>ai<X=J7FznmW"
    "BhY:i9,};):?8yaV}~IBUq4rGh_L}y$CoAAwkB~~9SdvccW2*Gl}eD+>&~>Iu}\"+&B(_9m><$)cv)5\"JM{/pNx:rH`qF\"v%Kj~>pPW?XXR{IhV,@^Kfqu(_~"
    "ZGEtzBhB$AA.;A{FcB4}bGmn3(aj?`Lc_Xyi}KMD;XK&(M9p!r55PWE_muMVgMUwRtU4t(LJ<Co}sB3LkB1*RWY4g*kVAG^|\"X*fSWFr}}HAJ]..ejv?7vaq"
    "^L&_Rt4}iLgAKV+>*\"lWdBH`Tq!WqW@:LU3an?_FvqFEFVyK.1v(k~BG0nYVd~2Lft]X$A~C_)u?6F=p7,!fn?]NA!>>hM4FKtn?e~(neuAMh^HOM/a9]n/F"
    "b|1[^KU\"7MkA(vIAry;XAA]_NZQ@\"FY1y#!MB^HkdTv(lL0qstzs?`rwQ%oBfH:I]lu?]/^h,r/hN([hw`{>6F6O*%tB@KXC]9AM^#o4_~o=TM#$]Ugz^K(H"
    "sZlBVLby?]^L+>Ut=}f)6~U4oVqufBoFmu^4T)K48)z(9A}Y_ASqJ|@VK{xDQw`lN?7Fpq><<VL]nZGjUG$k:X[t5E(|Iu`=f\"1+m?|E24guh~CB^hNE8Y\"G"
    "}CMxyKH1jtluq[{FQA{e4Mqp|bMA&kXu3M.>KL9)eL%^[@1uGTqW[Gssh2ib.xrv6hcGt2`$C>Q?]>_sXXvP<pxd!r5L#wn+b?=Jt@\"3Q&G`F`d+xKmGC[LB"
    "h~F_VLlu)A[q8W@V|F=~U%z(PQ#T{XiWn`gAhN!G%_Rq,Xn?l_m(T)$~yq/lskB_Y<VZN>%s\"~wtDT$i0wt~Vauc&;F)rf\"EUAIhnhRA}sASyN[z&&6^v>&C"
    "4<^SYF@l5*,>fqF|,){FPwPtO&kBny[9dB>c(_0}lWQ))kxSc)]D4FurrU;?yQ2u;4#_sy]CkN6GV|5A3RPz{Tt*AB5ypFZs)>Uq~}Xsp(%j1WWjN`2kb|h="
    "3K8leZ}gaM3[^%$V;`wqE#VCWMTNW$>e]LxqEvK2%_R\";MHh;F@QI@c=X1sBr?xEJ]~)`J6G27oT+>s?lqzn1*@F9p|GU,0|;c$o7?uPR\"`~zQTr`)Oj*b1|"
    "NZ9%\"FcisB;j>K\"C4T&C)_U|2W8}(_~CAAcG7v;XHDU4l!CE<VF?:{=~WjVLbI^,MAlqBw!WAG=ymZ?X|L\"vJtAATqVZEAynHAzLiM5r/5F_PD/o|#gLxKYb"
    "YLLF3vt(}=6~oyu(J>I`vqA*N>cFqy<U`J+>/v,$>jXFy_hq6(ULG}nB\"C+G:F_U_)#y:FIY]&;B,I3(45cN^|A*DAdvh|#}j_OD!WX?T{a18u3At~sWIM`F"
    "+F;XoBm\":y_g*PK1|%33|E`CE%Z=jA5}I@eG:V5(NC|~*06CyWJ(q1E#%*MH0tB\"O]TK7v?rU@^W|L*uz((\"@}hVB#V|fIJN^rdA=E5X@T[rJt!L;y|RRZ;9"
    "/Ck/u>{Qem5C*A%=,TgXKaCtXVCr:QA~lr9WZJ?{RVSBz/H!Qt6>+?[{+(pWXFC\"K,D\"=)X@tQAA8}2L]6du&CZ~V[C\"V@,_&CQ)aL>Caa>@$h]kf(V>{~>{"
    "IA9^j[NBDv(_ko|w%hH`JfyCx*XG41+(puwR(|.o@E3RT_+(})K?lkvWOXB~G7:)4BK>~C7W1Y%^:sBqq0~gXpS[n*:(p~jptKa.]mw$fg>E=mg1;4aLYitr"
    "q3L&\"mg`1KVFP9Epf?&B:skB^|RQgn/CTL9~l>|},Yl`+[fu8t|}inb|B\";E)|AthB}>=^f(u((>/FW+}=CBzq}a/av`dCraK>2KPcYTCCLHx|;)7YIPI[Jq"
    "E5VL#^HY!fp?qr2WSV$AbF}4:(3L*TpBOK*1W+()?QUt&P1>CBZ>TxVVt`ZiyFiin5>{!T*hF`;FryB>P)X\"@5AG|rxTuD%VNG?{T|i~?D(F_v]([3%`ybf~"
    "R\"6?WFUuKv/tk/{[/CXC#~\"`g1$Jq{Gr^yAV._$qOBiB]K6C2(/>R{>~AA1FPw1WDABA$AQAXLG\"AAXL\"CAAJ`AAAA<CuW|LAAAAgAuWbLyAIAXLAAAA@AAA"
    ":CDAk_#TJV$A\"CIAAABAAAlBAAAAEAAAXLAAAAAAAAAAAAAAlBAA6y1WhBk_T|#}J>,>~CJVAA=mTEz(o/xv>$0Y<KWF3+]s|FVwOt1W!Mx_NZEY+QP(,}ib"
    "&a)q*~yTbd11$$c4D`zhDcRxUSKCK&=&T/fw$$)58_]9U+,AaC5F/hSKeD<U)A.[<)FCy)oK_s~B)_uF`(t[JQZiX+~q(nyk(Xo~YL*_8q,Y3L`6WtyK<>x\""
    "7Tn?/I}}S@XQ!+\"y+r+>AqVOe]3)V_qS}hdG+_kZ9*D_kq=)R2kaWn+Tf)&^k_zxZuP/|BIAbFBtA*HACt[CXL1FSquWr?dGCA8ASqIAAABA!\"K\"8F!~o=B\""
    "(_gAyK|Fl_HAi\"AAZFi|&CXLg\"_)(AiqT|=>h\"RtGC+_7CJAgAVZEAk|sBhBE\"JVyKiAJV$Ary+r)ACAAA#TAAbF%tAA5FDA+>^QAATLV[\"vBt#((_y_Hx<t"
    "./>{Aom6BGpGAZRhXF\"CO0*Vb~!~9Wz?x@ZFdBx[iFAq?~ngvLV|J&d~2L};.r8AsDz_)M|F:11r/Vk_k3\"oii|~_.0Z5&L@ML]X9*:RI}mWLAy|q:Qx.MQ#"
    "#W}U\"}BDoA\":ULvlsB\"GtG[`LY])[|L0zUIc3v|Tr?W~ILdB@4RM%\"DAwNOZnP_}vqI*7?GB7yWAx?0FpAmEm1!(fLt`QNya.HD`GoOZh=xQ[nBt4}D\"2(5W"
    "zLBqoVd~!~L1Z|q?6L:Cw(DMq>/v))\"L_Kk|,}WX}KU4a|k~/_7C^#R+8>l_0B73BHEE?o+>(_9vxFqW>QJlqy{L\"Ej|sW@Lm>`6sBEAynQA@QhwuW/J&\"^X"
    ",XaMq4sEbLwC{|HY)A6F!WpBk\"Sq)A;C(XTX&_;vHwZqL`7yv(;A3F@Tc4_iB5#Ekg$GD|Z#EV$\"<)DMJ?wt)W%tzQ6v0ZGK!EJvs(qbG_$~$Fkr2(h\"#Y.>"
    "gJ}5v3j?Aw!rdB&_}>u(C+3@i[v,)3i`DqoFD7mn,IDt_M@W5y6CEA7s+~`>?>^3.Xr?y/KuH.\"|5EQ?,}b@/Whw[DmW+>.I0ZbXVHs1ZVJ4]KOn9B7(>(LF"
    "#Tw36FuA4MGQRz`TLL2WD_MR[qL?i|b|q5X@6FGYA*CFit&C(|I/jk]_z(}Fyn0ZT@x@{>~*7Yv(M|[l>2//p]UxsU9F4F~oRN~~G4E{jV/VBn6,FtdG5KMx"
    "mim>`N=TM@)^<v[CDMi\"Sq)AeyuZEM@KMf%$Mh=b.y^)/>gBo_qacVrXvs+r4A|x4}5K!_Qt&X#MdLPDNZ{q^923:`)k^d5RUsvDz`[ptERhuQQD)v_V}M9A"
    "Gq7L:sT&$)_/[^yCg}v{z\"6(.V;C`53(xKSq*WI@8Fo1F%ZO0)G7|zr<mhICJ|9>~]S|j|q(WLOA.AxnmB~=<?K1_)7<PQS?cB;49A5r^|1eswu(z3bRB_hV"
    "D(i~8~o|Mt+afGH^(}cLZ}FgV&}Lhq3oN*S?Q2lBJVnJew%$$hO{+Fm(+>[ETt9)B>h\"SqTL!S*K\"XAM.><CZYiK7~dD{rhBN?PD3(!}(_Tw_sVjc}\"vK&?X"
    "4AlBd~6F=sBtaX9~;C]C+>8FRt2(1K$A`~,ASquB4ABtI*xW|F4ydBEASq2WAM&GFKv((A9s`~Y~K`=~iY_Xu\"CIRVXR:vB*<)PW7v#T_)dGqC$TU)/>nLb,"
    "z)>/rfTt(}+bk|_)rUQW1_?zMhS!RwlZqW)\"VZbL_K&n(C)A\"v_)(AAA$AT|W+x?l_YITcd~qHjG:a/~c~[[?(7W+Wg52B~BVFUqb#+>TE:=tQ@Bnhkq6t*M"
    "1Lb4Mm^XtJiAt=f.09!W?MXFZ`j\">8[K)Pg4=>AG(Bz3fF~v\"aw%!\"PwaSj#AD0Br(`Yq_GZI5mL3_kZ|LtV1[/aK&??*kF/@JeM1|Jc837{pvS&|^1Ez[Ec"
    "+23s!szBoM^?0FVmkUI_=se8>fP/:NxqmiN[x_(X)7w/xH/oT4+=mtuW?4TQD|58{2AHbFUS&(k/Hl8qg}a?*CKqs}n_([}.Zq7~&|jB*h@PaI@QyWRFaI*u"
    "mio_#{(BYBp`8s)s95bG`r&C`~ML@md+X@M?U|QAC_,HME;XH`ut;)8AitrB[hSQyz=uHj>_41@r8Ayq/a`JeAIYIABtWZ1i]Q_qeuv3$A}T=N3LF[VB|L3~"
    "byPAl?Pt.$#A:vB\"7MVLUq*s_EL?,I;vRJp~i\"ZB7L@3E<ZL\"E)IZVIA/1OY*hj/aILB\"L$_HLOEZlq>YtGxOCx?Gr^XtYgA@rU)qh\"v9WPr#FX1m/$t`KQt"
    "4rqWr?~F$}=>C~7ylBj?6~A\"v(.Jm_<Cvr:&r>y[]mtfa,rvku$MCGiqVZ%hIAJ\"ycHONWfP5X^_It8*)MWL_xkL.AUHF{SL,xi18tL`=1<W_AfvVBa7^KMD"
    "uZBh#~{F*~SY4G*k}YD7[E,|;X.V;?(3icrXu:4l7XAYg_AwKEJh9F3[x}X@Q/|yd+/JnniG0Eb&,_:1*`v%dAN+@V7^iqV+DM+_SDP*4YXJ%qQV&*+`*q,p"
    "B@DG7CGx=S.>9Cc+ZX~Dgudu=sK~qyto=hhHa1zv`VH?qF1(.JZFcvx,=>WL4CJV8APA8AAAAA6C#}T)l_+LuuLA4IlB!2}%5pluJ>%\"!({L)>0k,Tr?\"FPA"
    "qW$\";XNJbFxqX+y?cFREB*hN^Lc2GATF#|Vr?XcM)[dEAYJ?aye/evzKADmud~6L&HYA/=cf(vDX]K!vdxN&MBW7y|Bi5~\"Fk#ngaR{C_~sw>L!{g*:jUKFl"
    "Y{bX`~!{gm)K5~0Op|JCi\"/,vU|llO)8?)@/T_ySJt&Fk_:v96/_<{/ox~D>5FeOk?k?s[WVD3=>Dz[X>+fE9z9T95|K~V.XAB9P.FgrZBjB~Fmdo=}Gl1%C"
    "u61estF+gZ}ln1Cz[MX`!p[X_4^KT_]v&r\"A1_GD;j._jqMxiu)B+[<)P@dGvQ/$_VDN7\"#MhFHF%a1KM:yndBtW7LNA@V_EX1B\"5K\"]Ly|9aLh~PtT|DAQA"
    "0AsvlBj?g~AA+>n_B\"2(#AU|It|)WL9CbclN#^6o]C@A0nuuoNfFss!W|LRL&nL/.@1)j|dWLvE.xAVJ`L>sXu!CKiaf_Xj?l_,4!TF*/_OyDxK>uV9sbXLj"
    "E\"r_2(XM0n[yx*A^*n\"*]2A?Z.9u7V)~hACCWGrvrpaW8_>v+GH?LW*|Ct,@@(+[F/IhUQMIKYPTF`F|`~:38~xTog(Mg01IYp&OV)R|qvO>/XJ]x;Naliqw"
    "zPx5~Qw$]nTvO`ewo}vU5R>~]YgBIA}!NCJ|HqClLAWCw}GAD|ydp=dG^__WhB5L)_lul=aLIv@Tvsc\"6&6rZL?ym+uuh_=viK8,6~s{A&~ZPLftuWRC)_T|"
    "!(k~f\"yC:>}F~C/q4>B\"4I7ab4)F]qZV4(JAryqW4E.yVZmL7LhwW})qd<6C4$B>O/R|*Tr(YF^k`~%K|F>{]XvIq>FIK)8*P@>zt}[h)M[yfZ~saGecv}R~"
    "j\"$F=~>Xp1PAJ?X_.}p+(A]XnL=BaFX$m2u/mD/oueuc0I%CJ5\"L2n]`oNphAzC|miY)\"y1^(w:(!DsEL71K=~C%4AJ4GY`ho?1xfWDTL5qC)s<0C>YFGx/V"
    "xLSqVBn?OLVtb##As1MZhuWR1nCU(fvRRwNZKATn$`r?\"FdyA*R>5K(|D/;EI>Ky?zui]]:yC&?L)>E4b|sBdFcA,LQQL2M|k~lgaya&mWD~w|YVAAQteB_4"
    "~EvD#}U)iBsF`)eLd~CD9)n(n>K`M/`jn`q^N$e}F_5~^XJ>WF8CdBJhiG<~;)]XE\"JV0McGKIt+P@!=>@cW4Ma\"QVK>h_<sBt@JA_y|4}g~h_IL+WWLeMJ%"
    "m/`pf&B;^s9&UFi_srP)g^;pB\"Y~h>.C<~}~zQSt!uV2,_qF0ZCC{K.1(X$A|F))OXC\"<)zUYGfA^|WLa4{{j4F>@hsW>m[FTq3(Z>}R8sR*yKbFSqAwDM3F"
    "qJn(c@dGV7#u8}|\"])06J.\"CKt/C`LX`#Qp:n_zqJA1RU|<)J>S?%NWf_vT/]nv$3<g^IIjni6`KiqlZQ@@/D|9+M&G`yq^:i+B_.yy,`>aRHF/`U`Ri8v<B"
    "mi){IC)(Yl#=Y](a]jq@rGbBAAJijV=>D>;CTxFhuWSqs``2{adD%S`4dGOi*WhB/?(_.T8AAAAAk_2W8ABAAA5F>%VB&._bt((ABw#(_V>_<vJt)ACtlBpB"
    "bFAAEA:CdBAAk_sBd~aFQt/$nsTF7I5(3rTpc1lZU,cG.LM3mLC\":v8A1[#}5Kb~qF<)vATq`%T)(_C|!(g~aF9~u(7AU_Bq7ARq<s.@k,qb8)aLe@l>pqBh"
    "_KwnxFl[wD10wqtY2KZL<y2>4:$9q,:L:E_E]*.MeGeyCVJtZW=m`X$MC_Nt\"X8Q[LovXoU~ZGOA<H!\"I*1WJ9n45asYJb/C%}s[`Q4C],J>8F@tKB#f6GAD"
    "E0V>Y^@pSqLLbRsCcqz?)^e|#T3D}AT|r*[E;s&C:TbHc<1sle}}]K^s\"ic4;Q2ZDvm/+tZdy:#_q>lB(&ChSt,F4AY1wr+TeMCF;C$@t>.F2(qL0/4Cpd]2"
    "o_=^}rO~lGmn\"vv%@:?~ZVj?K`{n2(5K<iP]PAcE,hd+TL$_nFNU}UZz[3It+>2F4F;)T@,>z|m(8A5v;Xn?DG~C2u&C/`T|/a,AbF2W,A8vIA7~9sV+9KJ?"
    "BA1W5Fzn/Cui1|8Fj,ZK#~Qq5vDA0>Otg~~}yn4}m?xQDDRw<AwAlNG_hw.$~N`L>s*BO]zKo1AAm_rvn(Mh~FayRV|L9F&6Qb!PTRvy?+#Io\"`vPX$\"BY4U"
    "\"AZiX/dXXLUq?tl1XQvw2T;3D_(_cBK2[?eDGA/VvA7U`9m|su:KS)j|A\"DM;(Mc!}X)OLt{OuIh)_8yPAL?:1{+9WsWODtuHM`~]H&`PjTL\"FK\"hA8X$$h\""
    "RtK>}FRt+ruW0Qzn>u/{[Ext@DlB#^5iaqd2WM3L9_#A+hcx\"L<Q]|4rN>5E,CI$nB1R?tnuqKL`yw?`O7B\"LFS&lu)hzk>)QA`G,RkgDQMo^dSFc_OP[`45"
    "T!cA\"+a~At%$A@5:upUEAMwKII4}o~G/NJ=U4($H6I3(7(+M=v[CbL%~gtT|9KeAuW=Vr?<s3rTL]K5CKq+>!~jnmW~~l_jq<)#A.C3rB\"B\"=seW;jsGAJrF"
    "ZlBB{B]Cs~3Et?qy|jbFBD}z+>O3h>3](Aj[6yN>I`Rq:a7A{kK&o=]D4F`s]L`L^nvW$}AFGr,(*2$~bI..LAx3^SDAI_rX!*KHfDmBUNQQgq],~=e~6CsB"
    "ggCAsW@@=:kqN{2?A^61WW8Mi_~F`%cscR/I*v(@o`:yu+^)z)O~y:sN;Dos}C*@0L\"p_xyW3LynUqW+>P5Fm(<)h\"oIoNRVSq))c)EBlDSx/>g^fvZt@4._"
    "51y_RtmPMC!WT7Q;j94TkNL?SwXV@jYR4Hw`kEd~0tog}jD^*b_s|)1!;C^)_V)MlIOBQ48GRq`UE@u`jnY+^4ULSqd/PM~Fa?|wIA+hkc_s7Aw}!rT/ftYt"
    "n?$~T\"t%aFywnjJt5LMi5C~UUK}C=vNJ~LD\"7}w/4_BV/h9~X4a&\"a:MnIs1DW4GR2rBAMJBCOgtW27}S|;)7}CBE|+Gskc~2F$DDfQ&O_}V7(`Kb14}5]bM"
    "nyYtdZ|F\"C6awNVL2|4};ABw8BrXCMc^aq`s+U6C`~S>`KVq<s1W&>7ycc_)g^?~4$?Lg\"R\"~L*koDYjt?U|Zt8YiGVIBVYsL!vhpsAtjLaI`)*&i?&|^aZ~"
    "VQnC.rwMP`9~JVyKWK5I%FI@7Mz_=Bx*g~4C~CIMyQ5v|zlqdF%q(,ZBp/=s+r>&@/~CAtNh._>s1B)AV_=()h4K`[ZqTLcG7~R*.)n\"x`,A?zEPk%Y^\"s=("
    "Y=iG&pAb]C7LdGIz?X]K1qXwJh5FODHDEAhtmWIAvMv/XMq>4ySqb46FTksBO>^}~FmWn(B\"7sQVn?^K~CIAIAXAhHSqJYtB6QPtJ*DM~LBD:y^|`W4n)84t"
    "i~=K[UkBFLgwS*,(,.Q||Qbsvb7o+WM)bG?FF_CA:4`TIEP?uqmD.Hg)u~D+.ME\"!rhZxRM/J*srs!|CTVV2,_Gtz&mL?(Z1qS7(P:Mi#$6u7>fO!+M@cFo0"
    "NDp*~]gn2(c~XQqkNu=eZFYCiSHXW\"JqS2E,p4p&A*(_inWZQ@C?_3%vv?0F6v@QG&Ud62Ati@{DgzC**pzK9T>$AVx?@@l/]&E7+e/aOeFH.].`<Z@>qF8["
    "Q)`G5L1TOCv`~FdB#|&?BAX<e#/3FuNy^G0|8/wM=(f1%9a&HGniXm3Mc^*4#CLXyMNz@otBthI`<sk=HA8CY4BB.vZVgw)?J4aq#f1Lr4B&1W1RHLYt!r8G"
    "$DhTmX~LF_lWz?rW%\"sN//v$!WbX,nhArWBNRw)v)YOQdw%a))eF?{Xu_@SL/~M&%hC\"hoD)I/)CMu@2!_MybE[h>)>v~C+>K`]K*~3M,B2FB&9hH(epiXW+"
    "p_gn7yDvl_PG^v?L5FPDK&<Ac1=%}ggKRq+QzggAMB.X[Rds8,GXE\"It#fe\"[a%a)_~Fz|dqA^SIaAIAh||4AV`6GDbL,?<^1Wjg`RBi9%`S)_c}&nlu1EgG"
    "4$1W1~#n4C^#yK\"vUuA&WR03@FX(}LxNqq/5IBfAmW.WrF@Q=4hA2}*O?(BAyKVL\"CdB4}J_;v\"Xr?$~>s,}_JeG<sdB3(%AQV+>BAmu<V7FoFmu{L|F&tcZ"
    "8Ag|OA(nyn4(Cf$A_s4M7Fft1WyKL?ay+TNVZFQtV+=>!~SqAAGoz|lug~/>xqtW8ArytW)AO/]CpB{ET|QAdFaIuWdBk_/FAY:>\"FnIAAAGCt7X?L@QRtIA"
    "Vd^$*BhN1R1ka&q3uL5C^C\"~I?B\"GDeW;Vz|#}0MBTZ4{G{]`Exqt/[>9\"x#DO5RhqNEJ~8Az,uWeM~19W#?>DiqVxlK(~jq`WTLa}=~Y&<g6G0|?}#ABDgV"
    "sgC\"UZ^XOW/C8vQl7z7v^sTX,?C\"f@p_x|xyIA`>oVtN!A3WCC]W*|:vIh\"Mdz#(~Lu>py=sjgsWMD+$[CMQ21{r{&G`G\"EtXGrs>s$}(_=~lxQ4]}Li2WQ)"
    "JA))F>AHO(}Y5{oB2[K&%20XtZmZ4VbFEqEpn?O`TtludNDFPw#}DAAwtWAA%CJVv(aF:vtWr?l_PAn?4FAAXL5F\"Cv(<Vk_BAAA6CBtIhE\"ME2K3FgAXLgA"
    "lBAAZ1XAgA$}kBi\"uWAA4FdB.A`xIYbLOAIAlhQAAA%AdNi\"fWAA6yQV4Mk\"!(HADt\"X55l\"1W|Lk_Pw}oh&0s9+ytTAuy%$\"Xm\"Ts4M.>g10v:@s?<wI*!f"
    "KG&HkuaX;`iAuut_h(8>0[JAja:>={1]FZ~VFb4yO$85eXw|xC4ZiAA*]Xf~m|GY[Cu{tFdB>>A~=v;%~~F_B_Dc>LR?Rws#K2O?&z_~EABqrcbj{M/vRV1%"
    ";)|K%Cn(q`(HOz<YVF~y0WF*?OFk>}:7`VG_Rt3}J>A\",Z6rt?hnE#cZJ?dF#S15zRDt3u1>7GX0cZ^X!Gj|V+d~aFknYVqWh~E\"JVRL!~BtIh_K9vu(8AAt"
    "SaxK|FE_HwoZ]eBw$lR~|E.vIVTL)_x\"aL#^0|7,/>`KBtv(aLk_\"vu+aTSWK4c~7AZip(<JSQ8v5yJhVLJFoV4ACwGYEA<s4}DA4F_sv(*_YIRVQ)}F!v_)"
    "q?4~=~/$Ik4M{|{9y(g,U7mlQ%.(\"sz,HQ~~}CzySC!\"<vQMW:7CM_b)v)<s)B{Lr?${Mxz3F_Oz&nwk~~cc:CKjO/R2dBy5=3oIW/Cmo_J?+5uWsn7F{(LJ"
    "2M$AbL_E6?#dc?\"FCz4oUf(~aIB\"<sxXuzr[(ME_vDeW(>`Kdza|GXRXMt.q[U&4%\"<V/^qF1WGvq_&h7,YBS:>pfBSJV**nPV4MpHj|4ah~o/Bw%v~~4LM?"
    "GuNNRLyK~$d]L>pFS&cJeAtuEA<p0Z0A6v2(k=\"H=v(v>Luc!|#}s~|Q5C:COOuQI19VZ=l\"Y&<hvX<F1WZ]dMz_AwB\"cF(k`~]XH`={M/Y~S|8y<B2&>Df|"
    "=o4V[9qyI&c~l>%qX(qK4K6vdWAM!_8v7,Th\"K7v+Vh=x}K{YSi3}=>@YnF>\",!{t(qr?E,CN+m?\"~aIu(}NjHh|lBU4Q?c1eu4}9G,HHY6Y6mz|sn,YXXpC"
    "K&q6RWjK3FLXBn!L>(AYW:2lmcpNk_gzb:?Am7d+M)aRinoxqW9LcAvuqVxtfr6>p_4_StJVK{a_bVEv+>GoCF5V6G)9LpDrj.]uetF*vbApo}WLK_E_vW[~"
    "$~?^{$;4:V{|Ou>SRR$~>GOZ|~z_d+BWWYI]C|$$@b3L#T>V:EzI!}mW2W+u4(vUbLqyWu)B~~![/,qiFB@_;)(@!^noRAD`KF]acNp?1_dBlBf~/C&CdB]K"
    "}v:C:>#A}CGh8Ep0r|hjoFRtdc*&=)hkI&^0{Fa1=XUVbL<vZ&Jhi\"0Bos)__~[`3}V\"<zG$vK2nCq3?j_[kprk~8F9vrqqW{E&|),c@c^VoOuTX~L\"y4}EM"
    "{EXo_yMhH`j\"+CEAX(vIa\"WBNhQQ8F~CJh?dlxX*=t]Kyt1By(|EV|.T9&6^c55gMY3F;vZ*H@{|jn;Uw?uhu{C&ItRQ9CX+yK4G([tuajsKn_4}Z]}~$qVB"
    "O&;IV|/og)y/BwR&GC*PKld+[2BGUw[XFJ=X>DQ!uK(=yn~`B&^}bvYg;f}]I;D{MV6<|F.zv0]41[Jq$A=C;U3vW]^0E~>g5Rvp1W8B)m#v}o`JFH1Q]6~N"
    "Fh8~3Ts#3EVGoTN}/:G|v+mKgG%~~CjDpR=6^C~J}}MzO*p],Q,1k#7AAi~yc={Rjw<~tKOWSxZH1g=iZFQjRJPLIoi$<AM!lxq+r`ZFcEcLv`j\"@J\"A<FER"
    "r3#]uwRfBhY}qs()V26Gdva,Ys\"KSU7PE@hG|KX+NJy>G_f(>KJ?<v`{R#R!ADC*MA:1r/g?](:7:X#rtHyn}.<Aaf!`?=E=%hqEXja@aCLVv(3SHlj#Z~n_"
    "{k]vAM~F&k,G7MuQrvkE{l6G{OT.4A#~HV&>s/9~AA|F&_n({L7~BwB|N5acAGRdr3h\"|T\"BDBwtHVH>*_0|.CN&cF8C!o$}xQ\"~NZZN$>T_r8q3j.OD7,)h"
    "A\"{kXb=h4iLPD.(,$M`KKYU@%A}}AA54FcQ@c~P|:)|@yF2Ff@q(I/%tMpK2xK]m4(7A4ujW5}cRXF{(p+z.Qt$$=VbFA\"gV<j&_w\"I<m?LCo}@Jv(:CYwqW"
    "V/5s3(iL8A4}(A}C;XJhed~oTtqWSQ&69+&77]3v7)*>}~,FRt(Ajt\"X=VC\"Z&8ABtsB/VeA2W;X9^&kBA.M}Fn(:5u?Y4.T,T%bNIKYQYW:E_t%`L9~,Cn("
    "_)I`;v+r)A\"C>~1K7Fi|#r8AAD{WMY2Q~11BGa+\"b*a>sD4LPEBAFqRS>VCOL}nuDA~1!r\"q@\"lxuN).hDIYNVA\"PFI**O+\"$C[VYKEkF!8AcGkB_0AG%NK&"
    "!<iRn[1u33/_7]ItH)ZM^3enb+iHC_w($hkL;C/D}0sBvK(M;>kU`EY(~ue@.I%,3g0G+nb:~W6F6ygY,vV:<vY(?)jYY4=~0g@YGlWZ,}OdPA0}o^6fryIA"
    "FAeu*V2|MZ0g.>}CY(ygnBMnwrFhxF.hmZ.MaG?\",ABtAVNVUG]_]XXLFB5y.CRtrP*[_~6(zK]|(XqWi\"~6\"+>J&[<sGfXLcsraPj|F7FuB5MEGQAtB3Fgw"
    "&C>>HAcZ+LJ?hqrS95;I6veu`~dGw|[ak~BA!rmWgAb#{L7~U|i#quJ?>{CcW|H{Un=TAk{EF\"bLTL6yv(=>hARV4}DA;XTL?3A|uu{LeG~v.}iLo_7vdBPX"
    "&A0ByKm\"Qt/VcGQAyKs&;p+T$A7vA*0~$~1_A*IAftRta+GB31XA2QCt!(_VPdBq(s}g]W8s@`k~>V^0!Z7rH:zrCvl?*_jn+ryW{Ki|SEXLe_DtzvCT[Lcy"
    ".oT?s/hnWZ$^|WCqO+p6oHtwf/y3&LODfxy?n_>~HAFHWO0B4}uR+09)HABGFc})Q?$|a^c)^E2sGxU@m~]|*~:>s`:ymVY|9KV`WuBCuQgt,oNhpLCDH*:2"
    "&_cfq`f4YLitv$4A>~Kq]&g_TtmWXLAAuWuKA\"\"C}A5FBtDAk_:C4}/>j\"HAAA4}1KAAJVyKC\"HAZF<sAA~FBtQVyKZFAA8AAAXL+>7vggNJ)A/D<AjqGV&>"
    "WL<~u($5@JD\"_J1F&|&C`~?QjqA*P@9A(,sZJ`p1:v~>P>;~k+;44L>p>TJ5M9P~x:V5N:&|WzW+|_eqL#s=K?*1h|)&aMbl\")C>Ri`vsB^H(NI\"PAw_Q*Q<"
    "A#;{hSJMo>k_+r4}qH~C6|_VW4=KFG,HVXf~Y.+L<>wnt(gs!_At#9a+o??N,.ms1Rmx?51i9AkxU4gF;C`sn(@L&NWBPjiImy.$=VQ.)kAAcG^_3CLXD`&t"
    "(Xb)r\"&v>X=V[{9rZZkAY(}~7~~CBt[&sP^3ac.@YRpC#rr8xQsC.TcBm_rvQV=VVLTq;X)A7C$$HAqyI*WLcGgt+T+>KLSq(v=VeA(,P@I`%|!({Lm\"mWyW"
    "&_<v0Bd~aFBt3r+>n_AA$AAA8AAAyKC\"HAAAAAZFiqAAAAAA|F6C;XAA<s+WAABtEAAAk_1Kk_aCAAAAA*DAjnAAzRe{HA1Lu|_%\";r.[p:ldq]J0|2(|cU)"
    "91xS^q%B>pmZ0Y//.]P+>JUMj[yCb)%GZLhVjWlB>CH;{LRXeD)vgN^Xet#}XX?V9tHwE*xQds#THv*^/13W.t_E&[~`0kKQ2[t&A@0RMJ,T4MA_?A+CBHv|"
    "z,PA{F\"CRCBHI45C33sQQt,)]r]}i1I.avn?CtSP(X7F*k3xI$@P2qBDAA@ma:LscS&\"k~9_*\"HALMXW:&D=>s,rAMCA2({L@Q7vv(WL_K/CWubLAAoV+>M`"
    "ZF]C#A\"vYVAA/v\"X%hN?Qt<vBC_K3FO^ANcMfA~~<`(_sBn?2XtvKt|L5LF|o}?)#_:vkByKJ?Ct^X8A.yHAVL8H?$<JAG8ybZXXL?\"1tZEMLHqI;Cr?$~Sq"
    "(v|LWELG6XP)$=f\"At0L<slB/&w/AtcBJV_Eu~X(6q[]U1dB;A(66n+&H`:s\"NY%HK)OYV\"#*jPC^EzudX?^Zq!f#~vqK&i?3KRq*T`~@P)>?T@h}E9~D0[>"
    ")=eDeuZ~%sStaap~,_c11WbLBBTA=V$A,(s[(_x_Qc<4TQ>{],(A$AO]XKPJEcn?q`LIT&>2b\"QY:>g#4F@CpBBB2>4}y3fECt/P!W(B_xVBzr\"G<vcx0*)~"
    "9{M/c4tWsDk#.skA.}0[yLwt=Wfi7>QG_~:(+Vl|2(wB[P<s(&(}$~5iU#I*YLT\"YhJ?cCiy3kVF}12s7?LQB\"qSHX0K!v^Xr?MV^_XWcL8G$pOwVeTL;j^s"
    "LXw(AqX+X##An(hi)AaqwB;/bFBA_Qyt,$h=fAn(N&^KE|duMVG_.vUBZ~zEznw}m?j_[qAYqWm_/C))_~B\"Qnm(T)B\"9~a|B\"(_At))B\"2F\"~E[OU[R(h_B"
    "h~S?n1&XQ)+?zK3(WXk_~vb|UVsKEn.(EVMKEnHD*t+\"AYCaQ@AD^,D,hS\"v+Wo?;AKt]C3L0_t+FJ|Fo?9DkgSWaIZ#.t:>mkotX)r@fwQbJ*#AG%Ps>(`H"
    "_W^LGAHYWX]W:FHcJQ<W/y+(&Mk\"F/N]{~wkxqH5AH7FGZp=/K+[tuFC)\"2$Cj]K!pVBkNwQOzJVJJu@vt9+)V:\"I&\"i+UKyhIv?QXj\"y?h\"!WJC)_1_tu{|"
    "w?9vfuqDD?rFV+)hdL31_%4AUq]XfX2MQt;)UUOWEusZ#}XsbWK&GVn~3H(k<)}RH7uuw[l_7vnr;LL`%q=~JhZF8s&CAAH7^ol)%~,[Bt)Jg_<^/)c=X_<{"
    "K||V5L6yW+aXn?R\"z?FGryq|j?[KQtHA+_#tHA#^Ct3rj?%GgD,uAYSWAwStgNsWRtPp*hsc[q}TQ)B\"CD+$qWY~3say9>i?gq3(kN{K.v],iLF^}sJYJOP("
    "1|Yt%tA`!q<#sU~~tFVZ)MVLBDfu(Mv?4`&sLAI46gcXk>IkSc~NPF$3{G#r8A8KD7dAuWM[0/m4sE+K&g(k<atuN?zn)K~6,WODxq$AZ4tW3(l_0\"AAnL22"
    "[CE_0_6`3IHBxTvT4MmQ@k)~Dw=?&[:`%&#\"$(=OyQnCEZ.5@JV|E/$(o`F_WB:JaGb`,TF>]^Rwky0k&_C\"n)~}dGBtvf+./C{~!rS)}I.]6tCA}G,A]\"iL"
    "aG}vdWXL9F1ktu,M@KlITqj?O/pI$$sI^(:fLsoDQWB|zSTj>QJIkBM41L.x=~)hoWOw/C/V/Jj|=pE)|K}/YAeS4n_!?c5^:mnBhi;QD|aSOjbKtp#rcs+,"
    "f_JqYUb~zqNZ_ho_eiP^+>;_7v}.or0QsLIqbL{K$q*}bgEOdC$ry[cFTkUZ0=CGGq_,}@ri<s/r8YaMIIcqTEd=4pZvj;D_a8m&+YV{4F9U6W?Jkn$r}~E`"
    "gt(T%hj^Uq~zsUeA6CJO$<7g.XT@3MHB4}M~0F1_iyu(@KR_NwUl]Qz[fTe?2c>su_]Lw)=~*+KCoW_;GuXX3d{nF~eu+_3|GD[JD?k_xF*[z@9wKq0Ux?@q"
    "g(2>M?4yJ*<)m`cAdZq_D\"m(q>EL*+MBVL,F!%Jt7EtJ~l5tCHotID:H]}^z!>jjC5r<%qb.aFx{O+UBi_Rt$a|;3KgtHA9~CDxyoYCA_)Kj0(+1W@b@EBI4"
    "GpPj/B9vtZQYcG$tFZ,faGjn@$]jl`g\"fLk_AA/VI\"cZhBZF~vItTLC\"j|@/:D\"I413?}F5y[)o=l&$niPJqURPA<@j&y;3CKjK?2I:C^L0Qi_MEHr7FnI,("
    ";)\"FHF1ENhA]}1a|rgdFV`W+VN+\"S:$Vp/l2AY;qdM9G|{t[m=JoTHKh~^e|[}(A7yvA5b!s..`voQ[|Vp,vaLN\"RNBHXIprjI\"EdD}9Jtq_Eq\"\"{KsyR&\"X"
    "O@xn}9)}YFWq?;[2yr/yPuiD.Aa&jg`LGo&CIjyQyq%,G7&B@mrv})O;=NNEmKiArSie3;iqj+(Y3F+[L:o6cF([lB>jQ@AGaSQ)N{U_YVhB7~AD:|f@G~#v"
    ";vDw=>OywuEY;`B\"BtfL+A:C/AU|gVXXTLgtcZD,5}<v(X$=ZQ=p(X[~WLlIJV+>4F:Cw}oBk\"T|LX5FUn{p$T.>J`RVY)[Glk)v4M!A_C2K|LOD`)T@n\"uW"
    ">>IBvky`UWD,v^L8MhH_[{SS$Mj`yw@w+X\"E0qyFOCk_Qte++TI_7vQA=?aF3uc4$~Tn4WZ~D\"[Cp=,_PtcZFhQWZIiF6r4}514Bc@TL0hUnEX[LsDp}8A\"C"
    "tuWXc^ovrsF2T}SDKA(L7C/}e?T:cyj#N~GBSA~~sB~sx1tWOWutl+(A6y%F~~g~ZI?%@JAMR_WD_);@#y~`Att/U[dxa],>inXu~qIc*>6a3vN\"JY[>NWh{"
    "#A5L\"yurp=4FpySqRhpc_[r#kB>WE|YVwA@pAYBVXFzn4}hNJ`9w1(YNsQDaBLCX*=KsmT:&d:<yBY|Xq\"AYfs&>?^6C:&6F4C))(Au~9Bn?=>CAr?$FTc#r"
    "V~AFq1y|pNt?c43+4=vdSw]Xv?4F/C`~EhJ`xqK&#}}~SqyC8A4FRV8ASq4}_~D\"4}NCXX_~Ht)@=A7d]CWK&kyy5*:VsL@a.@o/;FYuyWS)vtS&mX^|;I{}"
    "_)INuDQAPPk9YWW2C`5vl+1u$GcGb|MVphPAE}}F}eMER]G?k_JEZ~Q/iNyqK)eTNA]V4AvBWh_Kin1$1DO>DkSq1icL^k=!)7l\"QtM]A~f2E~LZVL[kW^h?"
    "F_?smEDjy`&9=r?)gCC|LG8~2/IIidhii.81:v]]p_L5R&73z/k41sV~.Wws*WV2c;9s`ujX}JRQ/om[wXkLeuY)H>z|vd0=z)cFJA!M1|1(w[3GczPt||VQ"
    "341WA>f=\"~074}0(PW!W]L$>i|:vCKP?p1;)b4U*\"v+r,A#qkEqB{!0K(M!Od~#smr<)UR9p[`jUqQz|g*:L{KOD!(63UR[pWun4C\"`Tb]l\"VW0%VE^_XWQ)"
    "|K@3+T9*dAnW`JZLMq~sbU/8F<VrJ>*>eA3Mq\"i&qW`L~C8BIA_h[C4Ax~6F`V.?$\"0MH`S\"&OQQAw_s#3VW\"C]`1u3Fyt]`%K5L:C3}e4bA$1>H$.1kj|xp"
    "3Fg\"%V$AUEXL(b>{2r*VdF<s=UFM3L=s,+tB/V:Ctu<>s`Qzw$WLBzoy$$%tUKaiH{zk~F2Cp#y?J_3F*~QVyQSD\"t&&_Q/F#W!:YFUw;U`cPBy|5`%iGGry"
    "jCQ)G?o1=+yueFQDRE;XyK/[)8g=rG:vdD]Cu}IM>T5tBFV4>N+>{P=suWXLeAJVEA/Cmu,A/C2Wd~hAhY)ABtpB^L)_Rt4}1KZF:C3(1Kk_:CmW+>BAdBbL"
    "AAJVkAAA*hCAAAqC:vB\"[K+>hqi}i?*_SqIADHxq_)HAAt_s4A:yAAZFQt+rAA;CJtL@#A&CdNyKhtHA)_SAPLi_7sNHd~1WV|%$Kr@QokT&m]g=QBenxKFn"
    "WCIV5hz|Ct@C(]8F@nCV:CSM?w7)TocA^BLXZ~qOX.:Ms:ftKUo[SXiN(v(3ZQxkzyb?nc|0=(zTGCUqUp|L?L~CcX*&TFFq,r>qEMZI>et[49Z4MW(v2bE|"
    "~lI}U9SDfrs39SZ1z,^|@F[nNEqKkAS\"C~~vyy_V3Fmq6FKq3FW|4T$Al|&C.A]_ra4Ax|vWiGDBYyatP@M`i|l+B\"jGAD<s\"L}E\"v_s=ViAIA~F4F3T[J5F"
    "]kvWh~%_gDS&l>$^7vc+=>$^StsBj?l_B\"]Xz(ULT|F/QVcA3(@Jt`hAowo_C|3rXgN/PA/VQKaFYV<s^F5vA*G7s/7C!TmWI>C\"X)$F<vEAFB:;AYyKyEqy"
    "?rHLZLsC9rd~T)Oz.(KT!\":phWz=6yuu+75A4}*V$\">TU@[KPA2WOQuqeBl~zK~Co(z((_1|uuyWcG~xV+K]jM.sWBF}N{U|G/15p\"+(9*5EJ4>Tn(SdCn%}"
    "p6.?d2Vw}~Q/sCit}BXd,[eWHjTRdzVtN>b_JJ))l.w`Rq{~+H|F#q0nAMZFSt[o2K]&$syaTA0[;vSv6_r1;sO]YR(na&F5$=LzEse+cFh$wr8A^nXA5^U|"
    "u+q)|)^!fAwW/CdBg(mFt+;C[*7M13?Qm+,a^3N0e?B:+[<RwM{K6[GA}_._Y+<J7Rmy,z,T;PE:IYIa=L%Exi4E.B1ZFv$R_uvWLm))XK;)!!m=7cZxAhVJ"
    "ENeOLCqSDzFE<$VB.I9)v7m=Z?nxvkk}Swd<2Y5Q4unO/5;?mI&*uO&M#cPxso#T.h1Z&YeRms!JKm7X&I^%:K*^[L1EuhEIEtU<[*xP^to7.MhM\"JP>QgG$"
    ",LtE=eqK[{V<SCaS3pvm.M8)9IU(uOIak!pEQlm{?/}$3YNU82K1KmHB}I_*B1ET5c#xvkmGy(4Z.YWU5vYb;T;i(IM,*hlzo!REAgBF.h]y&YORoyZ4,MfM"
    "%Ir@Bl(5AMBEJg3|aYYAW53Hb!MUmgY?3xhjIU&^U<&YGRBvUHMAPz~xViH7[{3Z7*tQ.o,!s=w)VIG)}o&Mo!3ydlQkF6U<MCNR.o>pKmh58Iw+yaHt+LaF"
    "/g._{EeOSv)Q;vZ.t=JBiKw+MUI\"VA=?LJ=.IYX9^nYxofyeEt3ZOClRkoJIOtcMhIn*5FIaiz<wBk3|RDZ\"e5kIQ6XDG$c?UEtp:GU9|$!*KRSqgjLAUzhx"
    "FiBF[{ZA8XUK`+2uHt:L_DuhF5.BeO8*VTX36VpZz)*Iw*z?&5h?KE!mT|w2E%H+TRKu3u;Tv5{KQ?++m=n!}D}j@MLo1ZVv;Qgi+xOH=?TJ;*5No=Rz8E?j"
    "#TTwU<7*4P4u`kNtFB~Jf.@QkGc?mE>h3D@p2ZRv@RBt+x?$^?>I4&+{F$e?Xx%k<IpOeOzY$T=zK1Q04XaIw*7H^/wAUC6STrZ4v=oM[I(+_UVj8cGE8h#T"
    "U9U<4Y[YKuD\"(mMz.wae=xx(R<_u7Q93Z.pZ5XqIn*uOn=Z?Dx~h*l?c[yZvzTO:{[rg;izI~)MU&M;Lmxei!6U9iOSvpR:u3eOAf?fx@h=F.B1ZmC:Sdv`k"
    "t=;?qI`+1Bn=7c]E7f7Lx27nSvZR;t*E<$?i(IU?UskGMoWztf)<hz@yqY)P?P*EOHe53H=.IYFT1A9*{Q6ww^pZdM1I;*/lHt7!>xHiqKy(T<Lv+P)gvm:T"
    "eMxIE:N/I,n!mxHi~[{k7n3YjSE2#2pZk5eKe*3D*^@n^wygsQDgfO6*`Q$&,!q+;i7IU)!!&5j!pEGjTDU97nZC`R6wD:Kmw)lI+)uO&M:LOxPklt%+8n$*"
    "&R3rUHKmDB0Iy/:](m#cHFlh%ZDgXAdMbK`,7|F$p!mx1oOearU<#*mQmuZo<$d5)IH*9JG$u!9EhiBBx25nTC}Qnx>5Q0CBYIAB8Hnz6cPx.f$$@/#n0*tQ"
    "Tr3e/MPB*J0(CKIaQztxcioM&^6navUV<wYLu=c5UIU(CKkGyA8*,SmwRwO\"@nUFRi5FizT<Pv~R=zvm<$Y5hJC*>G0W8cXx3ijj%+aAMB@I]$c8(M>LCxKj"
    "P?>c6navlSlt+xPH;?>I+=jnn=yAGC@Qnx#+P0mM[I6.fjlz8cSELkT^E\"JmEBFJ5+MUkGn?~x+kEMz@eO5Y4W(/gjoZIBsI?=3^Ia`nQxFi5FENZAk5sIL*"
    "1Blz&cdxBjGKizWABBgI`+uemg_ngx6oZREN}$pY5QF1*Eu=WBFJH*5Fmg=cmEchg<E\".MGjqLU)kEFTvA_Y@T4u+xpZ<i7I;*@I)5Rzlx*iuW%+\"$YvRR%t"
    "#2s=<i3Ir)qSJ,^naxBj]Wqb7nPC1R;x*EpZNBdJU*[1)m%cfx$h+]Ko7n&*HS>0>pt=wXdLV@~=)^6ciE`mKa@pW<1*bRkoRAQAAA{e<h!\"4CTV5*&t9(XB"
    "tK._nbf46A:C?UG>>J!(gB5K`[j,J\";(8CZU9~pIc/aBBK4_BNeL5A:C^UQ}$^xneuhKC\"a&<`>(0CiVz?3!9(au.K,?;Pr_SXVF\"q$@^QIi#C}U/=Y;n4dB"
    "aK?^y:2(6tCFuq@$W|)B:CAAuW;G^SY.=o,FWu]Fa2h=o|AXSE=hr&{XbtFCKUS&pIm4.w^Jo)jtY\"~\"cBDh#;t/$y2\"eCCUT|wD[_~DrI|(r`C<9\"TCUVP("
    "V)5WFEsgK:vJ3n5t#EYo,eaL~tXCh$hN*$)LGx$JC!3L#10\"cC1TN%S9i!;Dv<C\"AAAAAV,;D\"Z\"wXXIH&~=>`]_duChI_4ekZOXWFMqQ(N&f\"duDhb^0qXA"
    "/?XI\"Ue>.,acHEFh(^Dt7kO)qQvpx$K7wXSI@U?;%<E!LxgK3^~r[yOXSF,pg(v_.?4CG&u>9<DRKEFh>]Bt7_FCXFOqL&DAz_cB}gX^EAkAAA[lel@_<w,e"
    ",(dfR/n*hDQi6*M&M59ByQF5R@_9|tDIl%$gPfnYUPGeP+R7sX3HrP+u`QBR|AKE=feGy(*(~BTR=h=cst?wHfu&LV!1{uTP$g/}LtpX;H18aiBMNt?Dne#7"
    "G_)(ptBEkeV[cSvLRCXQbv)HAR;wFf+6:.:[{B)O&c\"Fv_|t?B\"7;npIunIB5I8oaS@C\"WFEHhZFmnuLNC7O=h_X~QHBdHx*^3iZE)gEjfO+8yZ\"{ACD,dfG"
    "ROXtQC]O^XHavnFBEe:&0kcLwAWCFQ!W[cotFB^Hh*%tS/pAEC_Q1pQjiLIu6cG!!mH\"XtcCVOJdupiL(w~H2&oIScx\"0HTObv,OItIuBeH(<&K!|u!P:g+%"
    "kS$ioHCRx1<v%FBuxHm6K&5kutTEpfD:yIP5>H?84o1kzcIu7eZt^9A?^uAP6kxQu_T5(Hi78cW9hR<wrGy+N?o|gYMP\"YeW\"Q{\".DXf?,o>`$[u`OXfyt+h"
    "}h`\"=a{:d#ZO^(ZOISY}G<QMkH$!WuYXKz<D7eY$xD:k_BfP#haFVq^t)HE9!e7!rt*\"MOQYDaxuOt$Hl9(PI76(2w6IX34eN7D)AE;f}Y{)|A@D7c,7/bzZ"
    "l*QPciHutnU58HK9HjB$3c<Dif2&13bLi*WP3dlBzI\"A|BQPUMHOLAS*MPTfmBK7D?}B2RooiNt\">BmP4e5.#[~A!H\"8>mZE.n\"t4H&!]q)T1tEE!cp`!MrL"
    "&Hn9=hm=A!\"tbFu+NPv1m*JP[hO5NHUM>H+9BdhsVi4wFe**V|Wt{WoO)ebC@JDiZHE!_kGt}W<DKf@8#~Wtn*uPThU38y(W6HNSTn+h?_\"t^Hpxvj6klAIC"
    "jQooOJ}QHBfGf#p1p_EX6DZhg$G0+W~B(Q]iK`\"9\"tUH@.@@tID)GE0e@@!M+(BC(QeuRQj4FBtII$L?0(y\"LCYKyiATNtBuxI!6wQ5[utmD!eiagZB??G9P"
    "wY:VKAyttP3hqSIO.(oHLQhddsY//Doe|(mUAi`uOP;h,!1PWAEBSIY)O{`$wt~DQiD,r1`\">wEel%WkM!]BqDgibyvB%iQHG91pw?JG@DEe))PJ;[ptmE<e"
    "]\"lU5(FubI;#f^!FB)[D:i&\"lU%4FuJH~*6y@vh*cPXZC\"&5(L@wAfE$AG=Cl*GOog}LxusX^BKO>mi~zc@AyD*d$zRjC?~B[O,jYL]h,tEH~x;vyWu*nDQU"
    "p*fL}>kHrLVZ2d2c~w@Ha&sCR<i*fOki|;]Q&WSCPRaS`Ej4EBDH2&*h=FvteE2ir#u8Ci>B\"N7X!5Vc[teJl195A<n\"{BnOeu@jB!\"ArI/*0EMOAX7Dyi^:"
    "S@`t+BV!Lv6R(L/wTHE$[jXth*nP`hb]D*n)1HE8Dnb]>hBBQI9*&AW|}\"HB]c/5muK!nA$B&#E_aS,_<t3.C\"AAAArSVx1w8(}w}J$)@w:[IXDQ\"k%g.5\"A"
    "{Hp9#7>`gROBrfq+,bBf1tzE]m,b;o.WmCl!q6c#OtbuBfM<:&z+HXtEeiHbJOvX<Hs$aSsb?n\"DIfE<CA9yS5iC&SAwI,(LOuUJ3(fZ;hA))P`h3eoZuXhC"
    "SS.wEH=A$AAA1[Ji.CrW;|2[mLluSLB~Y}=>SXzF>s4}4[fAlBUL=~Y}=>$tvF+sN~/>K?/CcW`~:`yniBSLC\"(|W|h\"kB.K5~}}a|SX3F{s>{DAo4luVL1~"
    "T38[SX3F}s}}<`>(,ChWL|$^!(kBUL%~T|vIT)3Frr!{aL=WFOuWAAC\"ydo$iA38B?iCfSWu5}Lt~D&e)0|LzWwYPP+kOG$5t)<HqQC&f]]>UuxI)4>8;>w*"
    ";D!eYoYx(i}H:U5tknZ/]\"aPQm/.QO%?{BF8me`E{WZuWdw)AgXtG)#OJnP^u_U5{HB!T3~)vnIuLeO+r`hcH)(P\"iD:JABtAAwQ)A,hkEMMm_kq6CUVIAw}"
    "R9]X@|%:{Hs.UtMZr?\"|X6{Tp~=>zH/aNJAm+FZ#.h$A5}*>`K$qK&aup_9vQYJV/`prD\"^3/FM{:jbLYkVZ=hFHPq!(lus?,FXY&O}KJLs_xGE\"&:kh?`/y"
    "2(cB.>wk>o=J@Kk_;soB4}3|rB1[^~hhl+!ClBbCM#%&;>H}=UWBsQ%\"vUEGnLqyRJDH1|9)pB>:fD(X_|TKY11+6?Z!?x3rUs1F(q2r4?RKHRe*_Ji_]Nmr"
    ".CaA/$t%_L?{0utK*V0F[owgl^e\"QJ,`3s9uiiXFJMKt?L/V;vtEO&@?@~Vxa2fM8vXw0}L>LyvanUu_zn?G`NdFFkJ}<4D\"8vxu$=(Nz`:Cv/EnVBo4,b|F"
    "9(3<6U+o|$hNK>St`rfPXLNz#I)r6_}y?v%q\"E8y?(pw*W8w7X(}^Kt\"r?+H`k0W1*K`Vq#(<`B:nluW+&Z)*E}`sNzFz3i&dNP?y||o5h)A8,z[BA3rosq_"
    "&|B*(A_0fuqW$~4nSqAY1~BwUxY)7Mqy+WXL6}_H+rh~M`~Cq#)<#A^sHjb~CqVB2WTL#|eVBc{EPq.aJVZL[n_%GK3F2[~UBMx?=~cE;^tc81luS+4~aIVB"
    ",fUK;sRd2i8G^shTa+~GuCHYsN[Dq`Pr,vd~Un1ui(Q?RyQ\"YSIrx`oWOKk_y=u(wF%$>uDj[XBt;a3(`E1kSSY@G`xN)8LX^/?z4SQXbF+bTEOvJ\"vW0}L:"
    "7s]$VZeaE4L%#?!F|Fx:B>XQ|HXzC7G?jn/Wk8eFhqF~@J/B719$#AznJV*)H\"[Du6zKb1dZ`BxRctTBR>Q?29]CdB`FLwyqwKh`T\",$.B.C}2=&Z}4I/a)t"
    "G_<peBZ~cF&|7v7M+AbEGCm_cy]XJVL`<s5}DMl`sv;v4AOA}sZFotB*/>[ExqBtHA=~`~DABDPA5F?~<)t[9e%~OZLjXH_H^Xy59Mq1F;kB^FTt}o/V5L|_"
    "eZK>5G7sxT\"T7}jkv(w}k^%kaSEA0F_~_t9~9~s/~&o_eC8|Bh2K%k5T_g;/^h4Sw*kB:.`*okZFp1xawuvFy$@DQ@O`eA\">QFLyIYJ5E`z=s_LsoEit]XZW"
    "&_IqPAj?byZ*#Mx(vq$+23lD+h[`5KJ`f\"&&>_Q~>}NJy}AzAA|=qFWu!rCHBDcaCfd~qFbxv}Vc+Exr$wKEz0^CpBZ~wtzXuim?;yeBUvHb^K\"C%tT}7vk_"
    "Ojm.7V78#f,WrpJVT{Op{1+e#fPH}IbBZB.Qqp7XwAJr@$T@~y(HMxeLBHm|]*{X?(3;],pB1RP|Z:Y~:BBq=~UV+M@q4T=h(^fwjBc)B_5F@o~@T:Tz^)[&"
    "^LnL}b|s`9Rt8s0AQDqPM9bGsvu+pK^R;1b\"MK<m^CkIR(=mgVR@NKStL~csgG;qPV,>}LSq.Fds7~#y8XWq9F;CHwB*NG3yTq6W3F3FD08A=n4}YZYA`s8k"
    "/_/C+o#K_)mIYAD^}CwF+Cr,OD<utdu?.>9~HAwqjnLjL?=F>!(<(\"9(b7ABAteZ[~jGN2lW2%>(n|N+hGdR=C.aDvk\"]X~B|X_E@D|e?RqF6C}wtOU|CVz?"
    "}FRtd(KD7HBn~`c.u/5FWc*tbQRG~b@<\"F0](va=5EQz};3MX^mI;s*Cx/mINuXs6~oI5$*5bAM~sB3/8vfIFA7FG;oZC..>5`IC<}KI[odu~}xqdq=J]>^0"
    "QAI>%_g}jUNVb]>(nu8hai1JU)8FaIs/0~$zq1&:%OnR%njX8kMc{F?$}t9UG>2(450LEOVxU@Z}M2;~lK\"}{[jqDf5FDr;)+CZ^GI#$M4{ERq1u8}jGut+("
    ">L&A_sn?D_7y+ubLZ#<sS&/>eFsvuuhB+>#pYA;?CqvW4ME_gtAY_)D\"NVmWmU@|`BJKbR7{CA`Jq_Aw=9hS1n|o\"NK\"[v/5kA=~)h._hA)vF\"aqTLFB9y_U"
    ">]]()_c0_~[Q*|ku&CaG=~*}v?^K|CRA&_:1WW*5B\";slu:>bM_>HA6yU_ZY{>I`\"~]X%O3F;CBAdFtsItn~Q/}C.}~Cp?OD4W}E[FaC~:T),\"6:]XcG6vUZ"
    "dg.>OAfL\"AWC%}zkS:K4#}7f`FlyWu5K6\"eZx~AHfAD7CHQ_b#`2j~0ICVd=ud.}H%kB+UvtmVHXV@xqdZDTSKbF0B>qd`5.Cq@VE\"rE[Vn/|FyFb@RW?qRV"
    "Y|,Qpy}]pWX95I]Cl~C^,[j#~&OWut=%W2(GLD:XP})_@p2}DYAaevi#*VqQII$bQA{3CX0YbF`_7xowl_Bw^s*VD#JyeZSq*NEq!uiD6Kh|2u^VD^7FVEld"
    "i\"Eu5*eFOwB\"@JTL1kynJ[c~l_9rG>1Q;v7cFJ=?Gh@b,XVFE|duNCJ`DqJtq?p_1[~`Mh4Gj_pq.MN/by#((}E`.>Gu[JrX:HY+W>nAgt{L(~znb&AM84#^"
    "u(mXYA/+`|\"RVrDu(KLjl[?YZJn~|I&#*&8~D`HVRVoQ@|2juu0X>Ceu\"Ug_8~vuNlA_[\"TsTW__Nbz[{LGyP|8IqQU_z,(T$_ay^)xK_K&_}Y}gAANBXL^L"
    "~FWBz?}F=CQAiAC&5hjnODFx[JW\"V+/V)_=sU|*>aF/FmWQ)uXWLE~BKcAc|*W]KyA5*$?&39aOC;`/43r?X1R;vsWFh8AHwUVB={xk/quxL8Ct|7*YLE|E)"
    "Q@iFuv(e~>:VI7F.eqxcQ{5vo%]X2^KnbrhF9y3WQ)A`3Ia&6YXX84xy$dCGi|F+iCCMT!HtYUjD4lU/kN3?Pt^vG7q_CD:,YV9L}CAAiGry0)E5xQDCq}9f"
    "S)B2t(L@zKBtmu:>7AkZVJbR#Kb,GCCAlZy(5KVt@*@JXFyn2(454Fk|xCJhn`PD_Xb@h\"Ka_@h\"jZJh9F|uJq_)(_.FBYhByQ(k$$FCE_<C.r,M2!httu.M"
    "j`*k,Jeu^Qav]91Vr.Q\"UZ{K*[bE4MoAW+|4d~^3:C15l?Et(B(X<_!~<Sqiq>~y)u`~vQctOunzxE/vb#1WXL/C7C$A3LUx5*F\"pSjLiGBtBY8}dG#we+pW"
    "k_6CJV4ML?N_Qt*[(_hnmBSCgAAA$\"!WQ@BAtB4}\"F8sAD`)Jc>T/}X)(\"aE*VBB9{)B%h~Lgt,$[>khXImrMVL/0_]a#M|F[N^X1WK>~CB\"@V7Lj_C&MV(F"
    "$B4&iLx/CtxTKrCBy\"ui[/WLnuSq9GqylBx1TL2[!riLx?x_haZ+n_jqIY0M9LK1a|k~PQz_]v}BIGd5dcP$^E8D%$:fzi=v/`O^U`}F5.3Tr`~I{Uj7;QQq"
    "s_(kU@Bz,r\";j>FwzaDA_EB\"/2bLsse_:qkA%a~>^KE3Uq9M5Ft~u$)Ajne+wIy}Rq^,b4`KPt^))J{KtDTq8ABzu(G^x!jDeuuWa)cF[9osm_,F#lrMSKU["
    "_B{26/?QtxdZ=JHL%vA5g^>~t/B\"bMZ`o}UVQK6FdEbL\"]![VraZ8\"AV@AjN^~BhB_TtT|PLl~{F%}gUe\"bZqKxL&kUETLrd)kjcoV[D*hAqwiSEZv*W|)F>"
    "\"vTZX?^EH7b|_)7}I][XRJIcBD2(sB(~vC_s[VqcM{),TsiFh\"iX#^NQ7,c~RL)q{Ty(;QSqV+Z+FAnI4A2[%,@ML`8DdB`~0RCtHAAG6C5lVV9F9sw(:vK?"
    "0nUVCTaMeyjE/VBHH}4$eLo>F|3(HE@W0|?(puv9:v~,Q@LKPw3rB\">`3_q3_J3(F|;{nUj.b13&_M?JdG:}WLF`WqtuMLiGjQKBau!4uvMVAM~GEk?G_J3L"
    "|kguv3Jb=~_)JC>O>pdE|X4mTqcE/20Q.kKHBJSQSsIY[)/\"!;F4,BEnmB~BOWl|OxGL)C$_k(:>&~iw=uNa%?<pK*K)N@.C#T4Aoy5,[V4)(k7C?A2kJV]q"
    "V/@~9Wr?p_G|sB*5p>In\",s%:biq~lmXQWvw%yHv)\"W+Z6M`EU.$L)2FIL.}FhaG?{`~?@!L8vhSN~u/At!F+C+_G4{TAA!yQtrImAc+63bKRt!Bcs5Lgw9W"
    "ZNGACsWX{EpvOZ>49A^s2K+_ZyAtq?p_x\"sB!\"BtFhP/lqtBpB>Qby+(DM<WLy1Wx[9\"VB=hH?JIMc<},COtIA4EFrk_Yh>QSqdZ`g*^Yy7W~G^Vv~B|?vO`"
    "4FtZ`]3QCq+UJNnPaC9US2\"FpC9WSA@toq&AmI)X:>9m_NlnNV$AEVGC,`?~o(_g~FUt#}iKtRw|)sB*hGUDTpEMaA%aK&r>+C(vy?TQ\"s3r(f`L|.#TD7l\""
    "xyFOf~qFOu`={LF7[vLV*zBDN|XYGiYh*%?)wV7y:v8AVL[owNh^/kA*Dk3XyhB{VC<WtfB|pij`{3^%M@)53]=X2Ks{;1dzFVIWMzvWNVs{<vFRx[~AFz)~"
    "lW!A>r4M9~av5:Ih#^&k.r$Adzwo[tBB~vNPQ)1Q*Km+BK6!{FAAiA&CPL9FxqZF\"+*>*O2W:>w~E|kB&CH`/vYVFVSKl_j#YJ,?6v+(SL{G<sIt/V.>bC4I"
    "9t0Wiq{{!f*%z_h(S>QB:vhtOCG)eFmBe]{m_{ZYMU:VK1m`sNcA?T$*ZYG|YD}7J`{x?.m}:XEd^Xn(7Ro>XZiutLOqWuCC^F+yAAy)Oteu|)dS)qC*T)PQ"
    "hw=~^L:\"\"Xj?D\"NZg8$\":);|8FQtCmQ@e~xD\")!KJXX<%P4APwcByW5R6ydB^L~FHx8),)[QCt;)]XA\"j|$$$AWL;ahN{QgA+>d};C3r1KBB<C0ZQ@hA#(tu"
    "o`h\"23RRb12T7?B\"T|GA$_\"v1(y?,GX_j#>O<WUul&:j:>0qUa`VA_*9DR%JD?uvX+k[n\"]U$k&AAA#^uw`~zk{~9s^)x{oHC\"js+_Qq|;]L%U,OBtkBq@rC"
    "]vuu0:7CI*y(c~#H6CnP_!x_O+\"XsW&n9Wc4!~qyQVss{=g{1W`+l_?ywrU4M3%qdB>>B_wAiLUE^[[aN+7ACt:&z:VO3uSXi_([WuW6G`4bK|,}pGL]FB+C"
    "6K2Iaar(i`Ct=sc)Ghwq%Cn?ob!D@QU)I?7`{$^$cFUqvB,$,\"OW_@M`S\"zY7LL1?}=&Q?+[Ou736Ro`b|856rQqVZgB9~/F$a||8FTquWy?,_;C{TAArCOx"
    "/2N.0[Dq5tBM1\"&?C~jnBYPj8!ZojE9KJ?~s?}@JD?h|)%B\":(ssBVtWh~Qt2TqWB`~~,oX4YF#qoVVhSKwA[>)%)nK|u?O@x\"Hfn_0Kicv?fGhq8)@@[Q7J"
    "5a4wYLo1O%1{hG8snFlZk_YF\"vk~4LXi}z|su>Qt4$*hyAiVbAvq))m(0Lqv,$[&YR)k&CmiPc)\"6>8FT1yu>vrKVn0EG5Mc#Azj/b]N{)S4N@gOw}Zt?)i|"
    "DHz?=>VLZ*5i9P?v[v6K0)Wh<)%[m`y[@oEAz|oFW2K\"C.N2=Vswvdf4dA\"a.@aR^Ksno}.?Y[gS95SRHcX(!^h0qv_~f@c5{3Wu@sXNRAqKt?knL9IV#L{k"
    "ND5Ks{zKgr=5q)cA]&(!UO+EzrVd*_Xx|4{5?0V+3KU!_CS|jb%Bw`:$yi3FeweEql=FY]$(g~3KA\"n(O&hAqSh~Lc_h#Bp%Lcdf*W95{Xrv_)[>KBoCL&l*"
    "m`hwgVdNtB^Nl/v8d+;F_rD?XMEFid;A/y=TYVPR?<S&`CP?qI\"XwBV/VI+)O28~it~$wAuC`~Nh+.s|yFl=b!Rn/l:>C^EOiAV(0n4r!KJ`Hq7SRVhF,LBV"
    "`JvJ,_v}3M=JayyqZBW:*_pxTAL1Y*Atz?$j2WgsZRGtlWH7!~q1dWX4`|j_Bt^LOc\"s%yckPd0nEp||5},_wv5K\"sczj0U=hA4}HvgAKV\"=PcB\"@`#}7MpI"
    "3rgB%FzK@FeBOWQt>r^qC\"uEY4tQ%qeBcjr/>NI*=>.F(_JwyKf^NG$}=>B\"Ct~YZ2EG>~Bw`&fA=WD@(_#q&Xz?#A5Fy:*_3IqC@)$=`[G.+kE\"!uD7ANM?"
    "nB;S4Yr4/yTL*?LtO+v3H^9~FD|j@EznnoY)dGC_K&E*bs$p&T<TO:=vKtdCBNy\"glm>>ND*{|0X=vbEYgs>!`1rq4zFvt#}:78=|Lz),}$~^k{JTv6F*|,}"
    "NtXF.{~.?&CF>{QAhB$kx}p+(>;CcWhBu&bvl_CC^LO\"eLb~_~MZk=,A[QxNK>a1P*<$zKe<F/p#cFcwCt+V7^jtf(IV%~%{_)C>d\"@`:(&`T|mWZZUQAD(,"
    "OX)G}C=W9ta}@qmWFAcwt>j(1Lt\"B[@PPtv(z*w/}eV|QVdLhzDswkp/L10WS&\"LusBtjgQ)cq987AzH2~y+$^3wq#(M,fo`,.K>q9qF8WbL6G=pScR2(ba1"
    "vW8}+^BzC&WX|RK?i)=t*>tvVZ93}F#{dZdNCA>rE$dF2nfr#r0Wqv5`1W2E[norO>j\"1uT@*>Nwj,M5.~aCfWns+>hq+B(viB3\"wBCGX|g}h[`EKADvH>i\""
    "`Nk_fGL#g%G`|}Y$bXf~][B*nUDFBtuW#AMG}DnLh~q~|TKXsQ$tcZBV6A.o\"|CBV_nu(}?Kry:v;jN/_>vr%Cp/I`.=bU_QzKMHNVh=51a|=&#Fe\":H,/8s"
    "^X=Vg\"tZ]Lf^g|Ox4*$~61Om<|}LLF_B:W&.7vaB<tpc3vr|OL$AfZv?Dc8y]Xl=(>Eo5qV>V@gqJYAY^FmkBYNYw!!|HYy5e#dv0WKX0Ek_{oNJ\"M~FESGC"
    "gK~0Mucg]QB\"<vy?nGJ1vB5M&ofAv33,(qcBB}E*D|AY%WfFa57,T)j~RtcZ=NJ`2IM/S2YL8{!+8A13],Ugk~;hgS(K5?JCcc0M+>bI8,;@xWMD_sr?n\"!+"
    "<53KgtpqkIpHP\"V+EBtDaSY)@RPGHY=59}D<|Rq6tBJ~[`R&/>1F;)v?DG3Fb&B*2Fg|v\"i^%k!B#(JBUq8UcgD?10C*cVm.byF~o~!FU4i}F&j_tv?$dNn?"
    "Un[`95%.1t]NAV@cd~.T=tW/5CcZ6(:bpv`W/VJnbD6a8}qHsyIV0}UFrsCV$MWQhA2>/>*_n}CC)_vyR|ggN&dD@qz(n_0Id+xi:.xnKVgg}}@ppd|L,?vt"
    "u+%KDHgDCt<LN/EOxol=H\"|$M4dR9vdB$<J>>s\"w/V6MxweWxtO?OAoU4K6CyaJV$KR_%C3((\"}]9>*\"a#MEr?nI?~0t.bk_;,QJ]ECz^,HMd~h\"X@`K\"{QZ"
    "UB~Rj><W7XwQZFP*Qt!A*~:f*V`|tu)@y/L1]vli?J}C`~Y~fA@QJV(~RwBtY)D_by+(zK&_2F#rTLi\"AA7~BtWuFOA\"Sq6y_J+\"_)zY._Dw@QM@/>%kLc.M"
    "p?RnQAY4sD4Z`NM{4CSd.tDGU|NE*hTLi|JqjbZ[=sM{6gW:Ll6(8}S(Eq7agB8G[^eWO&EAaq|L+>pFlW/V$\"5d(fM@g|~)UJ&_%|WW&&[K6IlZ8g8G&nb#"
    "bsMQfDV+sNbLAt.T=J;VAw[a$}q?E|.}c8`QOA05[Q#Dg(g[E`ICdEuW1RL1SI<5%G]t,u@4D~nFK&^)PQAwU[2i^Fs<4}:>9A9+h4r:6CiyxK<KuClZN*ZE"
    "\"sBw4Vi&S_q,P~t{6y<rAYNFPDZq1{F.7IlT4}$MApL\"S~MDMZO&;\"^)k8tK(FCw3WpMHo1BNJf5in\"X3M+>YkZ}:f&^7C<~dL~F.C5:vbtJ91bEEBvbgsKA"
    "UK4y@o8McGjtJ*qi4A>T/5VWR|,(X|F_=siq{+0QuDSqQ)yP[mBtRegGCq/}$5hGR|XVsU:_#~hVLLa\"uWEA`kQAf/<ve(+>](!t2(Y4PK6yyy~~}~Kiat9V"
    "eA.}@V_ESqJYTjv(.}<v_U|L/vQ*yK;/\"vx#_VpV}y;v^)|Ka?u(OLC_Tqduf)|FWnItX4?>O]8ZB>_9zHoW9&$HM1ndV>;K~}.w_Al3=W!fC>H_lW{&pW5H"
    "seb@a_dyFcj?tQmI3rc4~E3F/$z?$~#{:_(U/`qvVEJh2Wez=~FtU`MJa|ajXWhq#r(X+@\"C5dx*h_)Kh(KLXXyA0rRQ}CcR?XK@9C0snPp/Fltub)aL7yK|"
    "b;fQ|eo(\"?H>pFRti+;Wit<)Rt`L.CLqO&jGYfP7iL+_/_IA_E!A.J}}KD)X#|ULCDtP6&aFZ_l,HV:G4muBd~H_RkO+\";+=(Q^Xd=McQw<ZN&D\"{;E5=?xn"
    "{wE7|ki<aEDMZL~e(,zMZLpC$$d~</?{!rh~n>p1IVU)!F.v%COCTB6F`uqWbF,I^s?X>Q+[R&#AT|2WqWHB3F;viie\"A*p=0R[6ODF>)`Bqz#(XRRD\"I52E"
    ".IVZ$}0/2_4y:]~Gp>WW)}WF^N;s5h%=@~*u/Vj^0qoY55+VizWjws#>5FAb{+t>\"~@ze@6RI?RV^L35QqS|=t4(MId+7d]FO|JfZZ|9HFEEn3g~QD)er.HQ"
    "#H|}z(>/aD<))VeA7yR2F`7CT0O7FHnhU06K&AbtPX*~8F,(1KOK(9M/aL^L]nQA#Bk|xyAh]A9Ue6[WQDD0DASt~9|s9]`y<5/>&btyYV$V>iZsWZ!K}<:C"
    "?.FHB1n[j|=~w9fvDsmi/?sFDA:_\"tr0DYcGryUZ~~BG=v;n,Y#%H`H*NOhF3v<2lu\"MxNeu1+sGsy+(}B,=qlVZ/>r@Groookz/Ba$$2,n/XI<)dJ!\"))m?"
    "DM~svrcU.\"Z#DY9FV_ED2K(Fx\"UtMHIfvZ4}|K\"CIwl~P@^KvugY$_,|:a@@oV&n8)|4VF3I@TZ2EB+I,r=JYsBtg}kN<KRtv}LEk_gq_s[Z5LU[<)^)B\":v"
    "SEE@_E?~kZBVyQkqKqg~X)w|uWPAHowV[2\"L~soF>C\")fql+R&g~uKr_*tb!SA:VmAzRmB,VHogtxiVL^_=epKfRtzISaO^FM28X;}}=i6v+/tN`cJ_Ue?W/"
    "|C9kWjbz}IeWfL$G<v?C@gCA(${&](+vu}?Aj|XVl[hGBtb|p*ko:p_X>>n>hA\"+h>FLP(cgH`ay6B`V]Q:vCUcs%?g|~Fd~TK~vYy{Ly)$tU|B:U~G4x}|s"
    "_(>4>oEh9~\"sB\"nX~~PvayWLe\"zZ}4nGLC:`Cr(^Z[^,m?cFDzbEAA{>=~|@,>Oz2(\"L3r*[{)3(0X=t\"0g@zKs~fw+(:P74\"X2KgA+ri]VQ(|SAyKU|>$M4"
    "NFqvurJxZ:!t?zG}i~L\"W+f4[qy:NK(R1>P{lWWQPvSEEY2ej9_U,|#0Aw[XTv@J:s1xV&+PX[uB;COQm_}}>7n_L<Qq!>Z^5Fv+zACAA@4L%~b|5W/h^{68"
    "]>;Pk_su$A10`DGC=?j>@o1MK_Z1qaB)&LBt8ssg6m\"{Ls>2a#2|eW=Q7F<shg}sLRuG$d#7A~\"~p:8}YRsF$$00oV&k9,IggE)36#`>0#J]@~}s.`,69~jU"
    "yQNDB&(AMJXZHf%~Qt`w^|8~D_Nu;MJ`kqn}<h_KLvTqbUF`=~,$dim>h|VcpsI>>+eBuWA\"41yn05y/KPD&D75)&_\",g=hAfu{Xi_f|:vt1#zqI&,LAOGO+"
    ")*0P+~~]chfFh9r[^4u/tp>)fNT/dtL|MMWQ1kXV>>vQ6ycxtW+\"+u7(1KrykZA$)^gj#rss0P/FFZgg8\"QtRVcA^)rYb~^_vrS&9_$pAAt(;p`~R]8^(KsZ"
    "AM<K|xVbTLUWJLru\"&PcB|gA\"y^njE)AxkN/dOd^SqP(?|V:sy<W,,U;i^[$~=w/>~I*GC^Ll|RVTjFBa1H.q?)BS\"u(GzqFBt9K@E5F,r/h&_+ImW||aLM?"
    "#rAAD|<)#AAGj:5KN?B\"]CHA(_IVd~8FG[*v!X5R%_9)W|0/pFwZVC*?Nwvrg@/Q2[qad~s>6vm}GXcE\"v_)/JD^IooZ]>K/CA4p%`+ES&_EFHWoXu8<mGeq"
    "j|ECj/]>C&n4C5*k*);<3Q@~%(|)W`:[9rl%\"~QAr(>9c5rt>X~Gewc)q(?)~_/C.Ml`?~;,)h|9k_?}_J{EO/ZVM@!_PGk+*>GB.v~wTX_@NMeWz?8F[^=r"
    "UAPw!TFh?(2yl~$*|EOwW+Q@k=C\"bXm_Zi#WG2[?ttd+(Al[zB`~Z!q7xCfL,P*xVBWBi_0yZY:2TXc~(Bw%!FTy,Tb@iPE_(S(AFqZ|UlC~Pfn+q3(\"Iw=O"
    "\"\">[EM*=2F4o^MGij2rC7<#E)n8:U5SV0|?9@hBB?6Zt7})=)Rw(q?8V$3R&3$Fd20070g^{O%PaUfJ%L1xZHN,OZmorN,PMsH_5#X0j=nXaIe>4x(xZh*8P"
    "We8|+M2zo0Q8QkH6>nU,%f954u_$$@/P)e~gNa?!90?6*lkzP%/wxeF3b`Y04@RPggYD/mbo3007hh;uR%3wae|;SgyZ#j[O|h4DMt<!wH~6uaFgpAN6&OEe"
    "7q{cJdpHC6yeG=P%Va:d]$ao2=N6]Onbl_pzdo7019wY0(S%*w2fG6p7Y0KNcPZk:toG:!wHL6H7]/wOK,De84,RZ0k*wPMch./^bo70v6}g]pl<ea\"dK)wW"
    "2=qY}OxcO%N,yzg0AACAAAB$O2&tAX~DvgF1PDcOIvUQDl^argZ5{H6#[=L>&LMxvey!?JC?\"u*P.m!5v_w)bIWR[9,aY?Dx(go.eMe76t3Ptnmhe|AuaI:$"
    "$.7!KGExKhj;cy{QEC0Q0qXej4YMGIr%q68rjR\"DZhQ<zX#IFvyOUczqj4a5<HwR,^V@]_GxCg?,Dtoqs*iQgq@zP0+iPI7%D?0(7/_w1gi:}!XtmYMPfp`k"
    "=a*iMIiO>{\"xZ?LEEhN/8S1+wY!DkqpY(}zX+H:$S:^26/AE5gW:Yepny*uEcmmx5W+W@HJ&[*1k7/*tKgk}Q2MRDCwPqs#+s=Du`Hf*Wm%MNGPuxfo_[D[y"
    "EX\"EKkK`OcxXNIF%Zt<7YiPxLg/[<CXAF??Hm$;?1wBR(\"PQaYbyxu)WeIc(~*HaZ?NEfhw.Sqq|yASCc&c<0D7c\"DfgY@9#MO`utPaq~.ajDuPIu%00zWuA"
    "GXpQrl%g.hb5TIe#e+6R*Lburg2{|R|$EC1QAq=8;`.?SI1#+!]/vt}wMhL;\"Y1+0YzQ%n;izI.?VI=$;#~EvAs*iP^nOJ`CBBVI~&z#]J\"WIxfe9*FHq|<B"
    "xQVn,!%rvXDIE%}4?Ck!JxRIn?WkHwFCGQtr@(;$xXPIT%Z1YXuAAABAC\"CK!_BqM0L$yLU\":?Z)fzfWXC%5xGc#K,r>}y,Th=+`E_/ylidFEtA\"ZK*^ZI^s"
    "Vhh~AwWrAM$ml|6qG,!FB?to;$}~Z<wAm_`0#rt1ZSMyGB:h5}E`tr.2SRktMEtxO?Xf0a77rc&9LU*OaQe{5out3*Rw:ojLT?Jn~1VXq?7yvrR2?KF_w}uK"
    ":KGtJS=~kFfv{sz=Kitw?`~V!]9v3(|ss@0nGx#r;XU|\"};0y?szgqr?PW.s.}yW>E*k1W})X)+_M%}ZCM&|))^g=/Aqi|4MG>5kfYwLXFR|($F[p_.^f}=J"
    "pb9v@C=CxLgj!T<)Gb+4OW[/wP5o!uK>$~7sd_IYNB{B1XTLVFrm5rC&N>+Kn$~BV(vF>G#7tcEqaSo=QLRwJVcnnHLv2uz)K?etoVJhk>CDcOXXHG{>sZsk"
    "aF$nf+kN|FUoWZVNr?yD>`iL}QL<\"%&t\"A6F>G>C*~Ioz,R2cXBtJtqicG+v)PxVn?YSE|qL3~>~,}~K#^|I/FD,`Eby5$:&/&fC<),MT(cz7PfqS@{u)#?N"
    "G_]Df}+>@Qp?dWI)=QYLyF;XiGVq5`/Ct?+h@$!KAAhrY4GBK45C|LH`g\"/&u(4FHx{X7FZvDc|Le_CzdZQE,=e|r#9MmMO3H(\"G/VC|`X/J+Qc1+~VqWLWn"
    "i`%i){eqcxQJZFnyYqbX#=k_Rtu?}FQGNx4~?:8{<8ABI.gG;f]h/X6IG/BVbFmCXVsgAGL`kulN%~1[(v4Y|Xz|jpk=EnKybSHXQQEwluNhlGl3ja<@<Ekn"
    "3B@AFqyXF*K`SDn^tK3LdDV/X)WReDvQq=S(6F^XLXQQ@q`5|ABGjEEAKJfZV~4Vy\":T@W$qluBV<KHI1BHj<PIFaYh=&^c1MxbXl~^nexwYE_6y+}#AVL=)"
    "|#1MF|8MC&[KVi)_=oTL@\"Y)7\"wPyW\"K8y*JW;k/+F[}We2WinBVL>?EgwL&t=0F;sFZU@$_+1RVK]0XZI+$?P_GW`GYVB<QtW:o{LJ`Jyhqy,1W:paVCF^D"
    "|B;slKyAR|{#B^GB#+AT*=OJG^FVZ~K`yVX|iHk|qS9C2}OnS&UIv/<IS\"<K^EGORZbLGLkZ;jIGen9Wi+q?(_Itm?%AlB=V7L1qNc8<!Fx|l+jsl~2B$TK>"
    "aFw\"aL&_}FVBM@`KJI#TiWC_T|PYyK}KpCRV#>j_;y+reuD.B6tu;fhBoC<Xx[UF*3:F/}K?D7!Wm&w;xqJA$\"|zIMqW/VrE^jzL8l$a]Lo\"1:F&i]Vbp$i;"
    ")`P|i:\"Lx(0kY+?XU!zDc/AiaL:vrCni~B+y=U>X[V*nCXR=`QJcw$wY|FaF]C=AKIOZ\"6v?6C=);T:QS|it#M4\"J\"LP}yCU?4`IG>^UTL.>aFEqXv\"M1[.a"
    "n)l_5~e(xV}Q?p1|FJ.Os<#9gN<\"p&<JTF~CX+Is@F1BsaMB]Y7@`p+XXH>{t+4N0F~jcB*lT;Aq|(DMBBPwaq2?AF;~9JTT/J7Cf!*qbFg{nCTLj^!w8n\"u"
    "UQ+FeWTj{W^NsP;<u(Z]&8!ieH/4Dpb?0FoCh&ZuD>9?kZxI_Es5$FQ)rn}v\"MF5PL7DbP_@%MoyWmBOfh/Fn(P7}LXrl+CM4KH`C,x6oVPqq}BC+>UnL#!3"
    "mFY1RV9hS!,{|(`+)Fi_cZ[)T_Ew<)aLnbolW+MV_Eh\"d~.b=~luzfX~o}NxFOiAOVlNpByq{b|ggGU|L/UJ]Q9v1vHM/Pfz7CZ~l_T|WA|F;vW+{X1Ryn/X"
    "lB0EqyiSFteG0nv(AY,/CwQV]eY!,Fmu1KH`t~du/hSEK1BHkUvRwt1Bd~8/gw8BHTp,znv(i?UjNv}ra~v86s=s!vMVM\"j+F\"OETv4/Q|NZZ5IH(kfr[e0c"
    "fJt8OX$&$_eV;X[LZLr|?v9~,4RV<V$G5Ffu~Bi_hq$},A>~?}UVa\"7,pW7~htdWO>D_5FD\"9G6y,$c~}F.y/XEYk_0v4rpdxL;s+(Uxy!f|9M}s*F0|q|pK"
    "3FOG),=Onb<s*~x[jMWh[TMX*PB\"IAP?UnrB)*p>_6j+!f3F~Cs#v(M`EqVuJh\"E24kZ(A5nMEquF$iAI@bGJFXTcB~F!C0ur?6^6c%(j@5G!s1uk~!~qFDn"
    ".)fGPwqB,Lz;GnOu\"zr?,1KtP)#\";_j?$~OwAYRJ!]fA)}K?~ytu]>#G/v<U5A!DkZRV)\"d+j?)nBq^,6KNBuA5*C~~FXW1[lGh|<,`B8sxq]dCh0KS|w(Z6"
    "U~[_<+T)\"(byJSsUJ`qFNZ}sw}^N^,&Tt/Gq{),(JW75(X#}v_fwJ=7}JW`EfTvW[Fj6>TTj#%Tw^XurF_+sL|R>[F/wqr=$]LiNx:fU;`ZI&QsJyQB\"vZI)"
    "xb{bVBRh55rLVR{6)^Pq5aXLTFm\"3}5GhqhT5uYM=ss)uDvX!NBtK@L{Fqn}h()^j[L+aJ~F0$O+Sj*?kE[tOvwEb1x|<Ju)YF_a)ARwD|+T7Lo1JVz(n@rZ"
    "r:*>s(d(d+HX{GzF)awUKQCtJqK>/&Bq+r=2?QlkO+{L>.NAV~G`1IFBuu|L37Z&Vh;PN2aZ>]@L_^uW~Z3F;cd/>|R@Js[aVJ[/7{xr*<%Amu0t+_?9>)~>"
    "|Eby~./@[Q?18CGK#hbIGwn3/B]n{,,Aavft?v+bmf{.BO8_i\",TF\"7WfH>E8v{X>&|E<FB&O;WJ7GORF5l?dD{D?jBz+>_~Lh]D3s1rD)?KkKOB5uK`v$Lc"
    "mKk_BD|bet+^XINE^erB3CJ*<~!^D\"b@6#17$}^)b\"kB[NAGCA2iBG+k`sc)4G8D{s/>QXcA}?fGsy^n`Nn`jA<4B\"eZ4F#3]K[m&Fq[.J#pk,Hv+_ewKt/J"
    "yQsv+(,44^2L^X#vX)dDIR}):b2\"BA$hfTN$AHbf2(+3d]e?FEoUSMpv,FnuqE\"v+Q{~9G0e?gmuB\"@KAwS@4R<6|}_V:>V|^sVC9^G`$Yo=P@>{7P0r!G51"
    "PAT[|xR&EAw|;v0AP|s_p=ULT\"_~RQ4;Kpt*h_vwOw/>!L{ne+*&#\"_X7}M?al`s<VP`_qAAa\"rxBBgA#WB**_TtZdWCE\"2(I@m?y|(v`CAG~sc#B5LCBtxF"
    "FVWEinq1.A,l:sDYtVj|JL*<{Xmn%+k*a?m|\",<VPRZ4wdwX*`%Cm+t~9_lI^w)A!s_9B&Q)8Fku+C9L||2Wj@%MII!r7A>k|T?@G^h6<W?>+V$wuWn4TW!1"
    "p}N&$Fn]?(!3mnLAX;|GqFUEi1K.Fuj|Ze@EC|d+:L[DHGTE?j7R<1qq*5H\"CUfe>c1H~Vh+f\"*)ZNjFF4VE8<f_mnkBDrS:oyU|TXH`sz9U9K1L3_QA*~jn"
    "e+:f)_w~jvTL6_8CDKyW(~YFl+QJwQ#wSqMAtv0uXL2WsIrEn(k?+_NZg)?Q7v|b]TIQ#s<)4MFB+3k%N>=KeqhC[=k=x^@Fd=m?Nz*u0A2yi}f4$^myIAU)"
    "@_>~_4=?$?dY+>f}X[:Xrj\"M{p6,{ebRYIN|BK]@_3iB*<sBUq4dX?^ED|`G<<(\"v$85;><vwq\"6%^4C{T%$7GAwj&v(\"L{h6av3CO+FX.k%WFW|/y5iaLJ!"
    "w}zYE_mLB\"fLDH+yet!TV[kqv_]JkA;C>OL`6v_vb)O?GI<C=V2s/~[bqKw{\"y;~y?O/bf7C/ajBQ(.CzY*\"m(sMKn{vz,X@uRfp^9%{/O`h/YUsWAqE,TyQ"
    "|]G0PHn9JI]v`CTRXL_X*JMciq~Ccs<W6`vu=h;bfvjZ]*\"9{\"m(PcY4AAI_j\"}Z=/\"IWYl[MB0I`vS}GnD!htAA|I=T0IEGg7mt<ER/+C_ZF>A`@_^X`ZXR"
    "?y_%Cu;W(@,rQ4d#htM|(A4Ib#}J$A(C^XeMStaS@h/^wmORXL6EfDgt`~4ER|at2(B!$96S$79~KiOu+;B~Az!#{H7^B\":CS2*Bl_XFpB?Pm1&$a|wKg|Q|"
    ",}]J@~NA,^,[;X^|9mO$Q|9hN`rvtB7XUG5w?$ogc\"uuD7xW$noqQQ/?lq=~]jn_ILk|X);/)6<)h~*_E\"yK9~m|\"l%hNKNDf+q?vFgD[A=3r[k,.o_KAtV/"
    "+>jgoyNZb>xE8|GYuixPyqzFj(RQBAm6$!3CP4nU=J.>!rPX[/CUtu&>lG\"yy,,<c~^KO%.J?J6I6CNJYE)B+TLXyKn1PY4A{EjcD}aR\"s(CZ@0LC\"8AU|AY"
    "g~L_IflBt%t>6yA*;A&nIAk_p4iacsT/LyU^QOv/3~WGQ)e~.vpy6r=Jsy(,V6&GTnG/s~Q?_1>D$Y>Q}FaFwg\"A`Eqy}s!Au(@47}&v22mK`|(qXAk\"ZYnU"
    "3L[t1WEtTK_>_sn3zR]>~Xm6m`Pt^XOCYEIldWJV#L}_JFKCQQ!CXu}Nl`MG|weWp`q{xF:2cMnoUulsL:x|u+cUCAZVk.%\"eAU@gD{.Z`6F>9z,=h5EAGBA"
    "mAnW^L{EUHzBR9?>Gr8v7YI\"YV*h3~+[0rb|/WND`)/qg\"G+T)J\"=WB>Q:06S\"K_FuFm!(dSTHn(l%O/[n]`XX{Q)kh(|4e~fDG.V+hG\"vStcg/XlOf({L._"
    "pLg(g[C?.1{DVsS)mK(,NCS)AzF/[>a^b<8Z^|hY+4Yt>Knh/vaZ4MJ?NA8AHLggbL&\"Ats~o?B\"cBU@h\"B\"WLwQD0IA#F6C@aPjjBy|QVXL3LAA4MlB5kdB"
    "u{\"Fy\"#}K`Stp|f4wFewM/7MDB9sz,<MXFSAl~zW<v1xkBA\"[I[$eL!g:C>~]X>QvwkBZ~:R;C_XJtpWkHZV,2%~V|9WOLF?&|L<O>+~PqXr@A:yiq.LF%H4"
    "0B)Ay[{5%2@Kz{JYs~%_XLy|>>n_J`x#E}Q?D\"eX4K!yeBj?A~`3PAWKgt]XqK0E^_#}NJ@KADlr<YdFcvIAO{}IO+gBz3)6~*UB`F*k`XH7#GkAuK4E/CeB"
    "]>!FY1(|45uVdDu+l%aLJO#$S@k~JFrVEA_G>%z?;.+FNB~BUS2K(X#r=Jz_Y|e4%_fAv3[LQDOxNxgFQG0#(,WRy_K*Vj\"]9vfTs?e~84UZJpfhwtqq5hyM"
    "i_g1=~aR)[_swY$L^_}QMVI_0_y_JjCGBwsW,ABDvWSCBH.CSt(Ak|R*6X&GwtDpFYwQqFmB|LdLhk}}8V+_vAfLCA6C8MF`6Fg}7A~vIAzP#{$$WXbFQDyP"
    "bDAGw/Vxv?),utFp2a+QN1SV!Wl~\"uf+951QOzBtG>Czl_H^kY^JAt2>55a~HLhy+>n\"jWIXAGlr_%OC4:m_JXwM(Fgt>}?<{~YLyv3@,>j|zZ;<:?fA@)!~"
    "p1?}oi<Qe~<r>2/QDlnuZur.gt$Wk=rVOz1ZJVm_7IIwxBzFpvkaHLW4?TdBWCP9aC|{Fvs_T|3$T4{~aytEsIkAiS5*WLJ1ZVqi^R:~qPv?W#PwP++vPWez"
    "qd.4g~mr*WiWc}Mz`r\"L0E.lIY1WT9I4\"X;M{ECwB\"MhfHyw+roNy)4ylZbLO@u_$r])L\"PY*>0`z|K*rWi\"sZ55VEjHGeuK#~Az}zo~]X2kc<&f+>x|R*$Q"
    "l?etndZ+9GsG%C@sS:RwLfWBzRl>W/y{n_N2))SLyKBG<Zc4+_pC`~)A^6B&lN8Ayy=&QWnqsB~~Czl_#$1W$_fwbV%J+>,IU#Y@p_:y!}<CZF_>mZTXk>xA"
    "GtM%qCT&KCGM#tFs#*{GOns+:;/i8s$WA@mHQAj4#Gf[zXGtG><vKt0p+G/yK*o=2~yn<W55wE3FH+3Ar]T&$}dA7s&qw!5LUZ8UhBkwXAY}ztFUWCZL.[x:"
    "9WD~[qg+m?}[2k;v<VY4Tqx\"H?G|DU3YD`;~C,{LR?;vfFE$@?a1?T|L1LRA8}zKJy(vti)>Io.TPL\"EJ1Mxur1)3kLtK,I`2q6CkBB`knCt\"Xc*<CU{IY6s"
    "g?)R{a4}e~%;:<kT`kr#(5=(8|#Nz^g~tsR&7?{~*7tEK*\"~OG[t&|[Q_n<s^gY4`x;)JXMIGBwuiK\"K_9Hx{qzJ+>RF^YNB.FL&9t>JMJY^]Ca}!E]t/+f0"
    "eGTH=l#Hwq?oi?_R40sB~utV!nU&P)eF0|+rHjy];{5o+T7\",$Y=>Jq17c5i6G9pcBhBE>.CBteWG>5F>TTLt/dwcT+>YL8I$oLL*?1tkZj.A?gq^sz?1{Y|"
    "(581^K@6+~~BsWqFzyF*<Bu|3DNVWSEqlyu?D>!~YA6A2T3kfL44aESX*?y9A*7}?Q71g*lWl`tsSqc={F51$uT&FGz_\"`]L3R~[8BqBQ\"0B%pa{*CLET6zM"
    ")DItT};_p4fr%k8SH`<Q}V0Eulr|]fO@0_xCYX3F0k3C7|0~.FQtt*{/j\"|@Q?AD2(`JT(etNZN2~_XFmW8VM:[Ktunw{GVq!g=2,`!DDUa+>V2L?~<`K`vt"
    "@}3YAMkwp|QV1?6yo/Er?(*n8P4}rWIytBEMXFD|nu3A@@jE~N&6!~kx0@8F|1|.oxp_#_uWh~K/6vq|5K=/uy1BW+_Fe<Cfk=P@gw/,)[^KNfRVYg3FD\"}4"
    ".`,vNxV%^W_KCtv(YR;00r4MIKyA3f#Bizz#u(2F;sKVIvt.RAt%@?D[99{LiMYi?}!7&~i|5WAM0G`h_s[ZI.p|/CnKs>9F++i(lA5F`@tQx_;vt~gB#srv"
    ">]}J@q%F@4?QfziP|LOWTq8ZFhz?Bn=Xy(s9`K=5qP?PDqNB#fA\"}B_pAVvL16M+[U)_Q/)U03bKz_!rEYi\"{zm?*GS|A.d+IB;C`%0*EB34E|2?8~pF;v#M"
    "BF,h.$OTsQ/Ci*>r#H*0S&OjuVlqgn5JwW23RZ%h>KN2|YIVcG5L\"shWB~*6ZturmMnI9+!a}],s?TL&w/=j]1Ms;|N]mW`~?@(H}F[Na!UqYY2WJ`CG3Z}]"
    "AHjD&a5&9g?@3}yKU.AJ^,%<t@miFcVlQc1!2c_XZLII:;NCh~.CtWmWT?PAQXeMpC{%)M:>y_8BM@2F]n_vtBF`a.Jt=Vp?,y`T73|KGI<v_4fG\"vS&pWC_"
    "ZIkyqWA]UtcWBt8Ffqhq.ASt!TNV653L|oj3l_F4\"CuisWVn7v*h8M.svu$[TQ?50u=>J_Ky{T>2pW^nw(M*h4&A|X_:!ZU<k=FYg\"^)z)k4xyAM=J}F>bPa"
    "wK5]IVn(;czR3W7A#pNm6?I?.~4CRVl;8vfV>5BGFn&`vN9??mW!f(=Qrp|$I/3Q[t?}M~rciqGz:|ZE/;Tp!>;>Qq*u`2c\"xT$<vKSn4(RO5^?pw(tB7RB_"
    "{UbX5F\"FU\"s`mCvFgU{QqGIt3r0@qyA.oZCGS_9+CK)_]>Lsd6%>p1|Yd~}Lsy&st(4L$sTE}gEGLl\"*1KXRR\"dBXR7ya&+&YFn|2WAB%F?15rZ+z!Z16y`~"
    "AHZIvrRqJB,k!$y></,L%FY)Z4C\"fzH`fq}*6Hk_jAXXA_d~HY45D\"gV|;<?Mw+)\"L1Fnn#>+7j_%n5F!C*Pm[f(U)s?nI4rc48LU|^X)MS>Xl`)o}1E@k:v"
    "Djn`*|U{4p9~Xo*vx*]?dt1u4}*aIiv$t*ZMknc#QtkB}1O^<50~0k@$SvgFk|v_VZULXo}Qq?&BxQCqm6j\"Q&d+uP:NFR\";[K!yLZzY8]9poucgmQaFmB)X"
    "C\"Jtw=q/b<xSfL?.3e$}0AKo;sEM|FSA<4j\"ZZA[~)J]yaw%dGH4$Q>&GoBw9;IJyQ|CHt}lNVaIg*6>%_z|Uc&|9Srv~*F,r@;;euBh:9hq0WC52W,3D%|@"
    "]/IL&,`&TXRwuu3(!*x|v(G>bAvW]2q_;p8B#f)\"n(w=7\"JV0p\"EiDuW/BP/v\"n4TLbC%$gg{Q@~rBCCyQ.[$zj4R?_08)T@k~$~FE*`IBAGzc)$v?WL8Br3"
    "U[wE<se6zQ~~PAzRkflBk=[i:F$CEAr1n(r?p_;vc+%>4~QGdZn=*_~C1B2KcGfqgty(h\"cBb@)^ryPAC\"3rGCF`y\"#A}kHAbF4yVu/VZLKvRVd~IBEERtL)"
    "}FGqpVt[|E;vEE*V)_j[5C$}J{[NJVpBk_]kW+1W[PZF%}YZTR7vAYHXHBztGmM)xW)k%,L),go~\"%A<(_;{yFYjQbaI:aqi7G;s]9HvP!rF(v%DZLhyht_J"
    "r/JIAte6*\"h#oBfFaF4o?LP/\"oNZ1K~F(qyt}Nj,P\"2TSW}s]}c6fG6F[Qf6@/d\">LJ`>~6xGXw:Atgt~+.hq1x`Ph}EX`8){j3FK}AtaXrb`EwF9V;?]Ky#"
    "7M@K\"Cwd)J1/m_Ex0M]QM1KS)ArF`UG&@J;v0r_gP(:vduv?>;{k+W4}BGvtODK&c=(o[`!fK.<s4F5*ARJ_X#h%(Bv<?rlN$_u~>GmNa)KCn;s18A0dYk(^"
    "([7Z9ruQqCOZJM:W?nug|XA>CR))*>Fob7cWZB[Z]tBY6Tr/{Bp:t:AN@mO%luDh5IMx>&u?FNuTPX=J]AvUo&wtSHp~ZilF{ul=|}\"<\"Cd~S`tnVC_b?bB|"
    "=+V+P@W!wthqlAD&RERL$q0y?)f#(wQ|Dw[)$t$`})IW(_5v/tg\".}6(ZM4Fuuc~|K=FRVe+,>)3nWo?.?X_zBQEzX+7Jt]@IHdD=Xau!G>m]F0(j_hwi+%K"
    "m_]>B*N&+BoIj|1{/Q9paVg~tcCtG^JC.Ap#ZKu&=CsZ=V|^gA0MDV+yxSrTAM+x9)?jMQW|%SAj9}?mNBo4v>(|r}EJXFw|BA]KvDTB`=XKP\"^J*P|k?%?,"
    "~QOzx9k&X#/r$};L~)sFR=zw8GqCR*M<P@~rnr5t{~`N&CFheG~[|Ge6<?Vn3Z1V]K%qbSB)j_IDl/3Pm>StIVIA{FK&_Jy{pq~Y7}4Ls2`GL&E_?s=)O2OX"
    "=6rVli*Wi|Ex8t:_+_XIn?l_sIJq^Xj^NJ3}cs,b:v3ruW=/YFk+U@O9k_n#?}GBr`4rh=XM:[z(cB&_!wW(_4*GYC|rfg>}h|{}\"2}~+L0)d~[}q+9)U*o="
    "Fnp*,$cFKo3(23SQCtnry(aFZF{pYNwQqC6aeihn]ndB`VMKZCK*hB6Oy\"7rL/h3\"lbj#F`RA\"q(M`Eq[q9J)>G_XYDjn`[CSC[EbF%nh\"!G}FA\"O,DGT|ZV"
    "q~/>b1Ux#Y&_Ocra1h(ZJiPdOX1W!HmWuuNK<C$(Ati_4yC#A<BSgq5.%Vb=[HxAz?$~htWjH>wzI12$vRErctNJnb+k*XbbldVf?$nwhSM{>Wm*hX[9])Ps"
    "TL0_7_6$*`bCoFmiI=_[=%BO#~DOJDlI6G:y08+(ABC[0XL@}L71HY/&`WdyYScsrPX`&FK5)<pyD0k=xQeCS:i#qBF_Q+%mu)SqmugBX*|y/]f)X!6y80*>"
    "W/5yhd5W2Gep\")7MVFB\"%X&>ds|FnByh4G:CcZB~kFwqzZcZL/{u:oZ%5FK1I+U~yF;CYO:VBH;ycB0kT;WI;]Hc(~l_XAt?MGdZdN??^nv(!31QwA|X`LuD"
    "GVF[Nc)>T&/C[Xy_cZ7f:_*tMR33yRAwwFLM~~Y]>(ludFdcHAzFNtS\")\"]XK>O/6vH(~>\"F1qIA=J,F4}jgx/0kPV_@6F<Zd|cN{]WLD&c~^Ks~\"a^Lk_=~"
    "%CiNs_et8Z_@[K|Ka|@J|^#qL|n4&A}wyK5~>FqV>&4R_K@$n%k^diS&`C[RZ4Xj3vP{5F+)XslBhJZq#Ya!m_AwQ@YH2k\"Chgk_yt)vV&dAXW:JV(bC~D/O"
    "v?2fh$*>>%][(}iq2/0{EsC&}Fl|;X{eiF,|fYEhL?9~AAP?r~orY)$L)kJVWC&_.C.T9h\"Gi0;XY)FcKoSqX?1}`_,$bX&~K1E|(ATqDVpBG>EAQ@ULdt:1"
    "1OLPN2D\"iF,|)~uiLVAA(LwE(\"S^^Ez|h:f4.=VqW^zLPPlsSqQ~mFLI[.$}$=?~5a,YY@m|kuqi_!~F2(}V&_PA!f??j|5,P@yRsyPt:T%^&|g()h7F6~lu"
    "ok??aCZr}@xPRt~9y3rhCGq,Wj>X5FRYx*&^CtcEW+H_AD@$`~q?V\":>n\"cW55,bQAkg0:Dq#}Y)FA4$0[E_=~>r`~M?qFjB?L{K?pg(M@J>ss3(0*]VZFB."
    "`~q?A\"uu2>gWTtl(_)dS?w&vQMnb!vVuWjD`6IeuN>X`^0s)e?H\"7CiiR)*vX7l%DA6T=V9~+_[`qWyK)_jn!KC_BwdBd~0JGLxWWS4G;v8Baj|RCz!rZ~&H"
    ".C>Wui(hQcD&mXbFOwOw1W3Fr~CxZN]{AzvZ*XqO{UuZxB?L5G>5*5@cBzYvt9FtnOuZ<XDHE:S\"\"LizxzqOMoDzIwjYvXZL[$V@`LxIorp=Lo~yQx]S]{Bz"
    "Qw!X8KaoRADMAzZxbXEAT\"4!IzYvyi]{hAf@aM<HK9HT8!AzI0yi&+*n!+*YFLwWz=tBMM{IK**5AMfGivfbk$Czs+_WZP,RuZX@KMDH{%)MkzDz_wdVFtd<"
    "#+>YDJE\"nzw?VzQx(f@c*nuZqYYM?5Qmhj\"L(Hh.nz(Ao+wX@S>MV0(5^L6IK9+mjz]y^uIUV9H%6ZWZqMKIuZe@`LXKDCJ$LogzC\"AAAAxQdx1w7WSukJw<"
    "O2*$0tHQ/jiaEt*i{B&7K]UE7W\"wMg4@|r#1Z*[D.qEkJmA?II0zO!p.^>TEVgF/=p6_DX6Ceq%Np+GixH9SUs5kiLRuugF[@27[B)UERlneh+~taHFR5V]/"
    "Ot+w,JY<Y4I\"O5MIM7BtATTBlBAAuWcBjKg_,hpqFCTFfp:&DAQtduxKk_7luL0*rQKqF%DAOzLEyKS]Dt]vDvNFuo\"&GHa5OI6%e>c;6cKE}gQ_8sIt7A6C"
    "_%~=O`DRMxqKd[KiuLIXcQCq*%m#etWI=Uu>i~uA0YuQ/pPz~2;WrCAV_<9}Z?Mx?=C\"EAAAc6JtBAe4F)]OvjZ*HOWA.t+dD39#4n|(5EbfW%zd>t_BF!Es"
    "I,(4?whe+y*HFD^urP.Qd/3W*(?BdQ$4{x}9FuRIZ9ZLAi`ulP4iM?+M_A[whb\"&(tz+j*.O@k!Ji4|ttHv7%t\"xt\"k*rPogK`d|S5+H:9LPz{ViqwFfu+FH"
    ")Wh*iP<egDDAa/@tMIjjKIz+tAMH{PjvQj5WAuAJP#Mc5kvt.D+c$@FHC?FC5Q?zjzs\"z\"/BwR<k!*\"(.D!dA)df/k`B8PUgbS#k~taHc!@gkzW/.tceT3(t"
    "M7,(UE9hAQJO+W}BuPyaB$r_BB`G9%\"yjZB)nD.cS:KV+WLCsQsk4rItDBjI^78M@ClYuPDhGBUV~\"?DreN%y@i/>uyPpjVEgZ)(~G$80sq%f!>D~eW6m0.k"
    "q*yP8SGB.5T5hHs9*p_j)4^ARPdbgD.5(?}B[P:i}LjLLu{Hz#4}Y|~(qP~j{BJmtLMH%6~t$ffRjwDHK!(H^Q\"(?P(f37=vAi{BG!_s^25(_Dfe04&ao|^B"
    "TPrZ*x2kCi{BFRZ1+hY/\"tXHv(|x_T=B)O!kT\"T2GG*D$db#k_DD=B4DNlf+(F#(mHk89ZKVt\"}uyDnb`RAR)(2HL7{mobWc*\"wEKi|7.>uL}B$7Bd`E6(<w"
    "lIb,,7\"h_BoP7bHu{)*(sBj8bH};4(@DjfC+_:X|`B]OGeSNDYp)oH]9{u/0Ui?Dben(DNtLq*RPhheZqg&?7HL!ks~Ef![D.df(7*A?gYVERf]\"}L[hJB(H"
    "0$UE@v^u$P=bY;!MfXaHGR1xCt|(/Dfec)V3`$_BxPXh@^Im&?\"FaHPLZ4yc~DHH&!i$E\"~>*HX9mu+t>n.DYe?(&$5_o*GENh?^?JXtECrR/lupunEB~eJ*"
    "XRQcpY2OHju2,5(WkH0Q=x]JhL]wjH!!H79I{B%O|f_Eb4BiLCLRaS.7%4(D3IyuAgon&BHE}ki:%TsL\"B^N$QYXJz~t!fz(yqR<\"(UN<gW>d|&WECiO9pJo"
    "[hHBKI$$,74[/u>ORdCdlS`A1wsd$w0)?y]BUPEi%$z1%?)H+9Awi~hLOudG_$cMyW?uLEocLCgZ8?_BwP/19<@hDuHH*%$AaL|t%HF9emviX??w4eL(4!sL"
    "g*KPqUPG.5VMyHF9BtHag!:D7eq&sCcO_utPLh]=G<&??Hc!ptbeg!#wOGy&0:y+/B!OZVdZ<aT5eC13zPsb1cpD&ec)j=`$kYaP/hz:d||A?D%H2+EaAiB)"
    "zD~hO58:#?AH:81hj+st?w?eDz#~EDi*#PjdWed|6WtH4S!ex8rt+D8G!m\"YtLqtmOmiaSeEsXEC56bP*$)FABmI2+6yon{WXEjhBNoB[\"[w~G>uUEKR^u/O"
    "njFE<Cq)UHF8N5=o&L=wJH9%MCo|f*ZMsayd&T?hJAHdKuGAEAAAYo=o?_,\"yP;hzH,MZtIC~9@4qB5/Ku0aM)SQ?FoY\"Dsg+[]QP5@HF62m8rHG@D9I+!1["
    "c4l*wP\"cZY<Cp4cCb9qi?&$LMu@d+:.u_Qj*mP?hH>M\"~(4wsIjzr1nqF)uO<eN//TU5LC13,TcLhR(wWePzEa4nVD5FAABt4F2s>{C^fAmuQL*~9~1W#A.C"
    "fW*~l_ERluUL[~d~f49t4F$s^|#M34,CgWAAZ}Zq#A*CpWAAl_=>$t2F<sY}W|2L/CiW1~l_QtiuPK5~^|2(7\",CsW`~Z4`>kBULN~T|HAht/CsW,^[@9Wau"
    "wK%~]K%y$tHcBtAAAA79]amB(4^DYe3;G>p_p*IQqi~.7d#?qCcRh9R@4WDu,JD71XjZL)xPpf3OcS#?@H!Q#b+hkLNB*d<.aY!Is*uP!m=Zy(x);BpRb#6F"
    "2/HB&eyuK.;>[(XP\"inemn.(|H6!}oc#^>BuGH7(E6+$}uiP{a\"*vBABAAEA3r~L:vH+>HeS.CgIJVHJpFN+B\"oP|II*VCk\"GAcGht[`FOg>BqlWv34Lp1mW"
    "M@LWm_ynz?G?{BVx?LH`HPg}VCZLG|$}t?#\"))>H1L@{[C{L4}:I4WaWUED[}9V~:J\"`[aWX>`8F#}#MC_PA#KQE_;,r>S1=qy;rCf:QkAskeF;<c#COs@&k"
    "0FC&FG#1PB3w[)$C(~M=$~?9`~NVPb&t0x<kc\"nA|^JL417KAG;m+$^jZFKJvTU)j_>t,w(jj>[DduTE.K>m0B<hd$SrjBJJH?k[H^8YJnk$MZ^|l_>]wV+K"
    "1|i_pCdZhFitWWgg\"MI1`~QVx(d/2u^)/C/s9u+$<P`Fq#mK&\"Ou9KsW81F_\";5E#v*+eL|Rgz%a?Xq/|[I*4*&~by)2A1EGN2^sgN1|Bwe+A[I`dA0Yv(j|"
    "`UggpBYI^)eXo>3IynT|l_Bwvr0M!\"3r^){FUntB?L}}4FC*#ACtHD!CiA~`PXNF$~}`NW~~Tqi&4AjqK&#M*^\"Ca|C>=\"r,+rVLqCScqWVLxq:vg=3E~vYt"
    "l&dAmuPv~~7CmuNC+_i|d+)tk\"<~\"LA\"~s$}{|+^YFq,Oj.>ftz,}~`Ez|c+l~BAkB#^1Km|g&B\"l_RqX(:&$^6C?T5*|Fa1ab^LQK,kaYTX=RPG!W:C}}~y"
    "K~1K#B`Hv+S,BAm_]X!Ac/>J5LAA~=1KftF~{Ll?.F?$y?9}D|]$:>4L0|4ynUY!5ILq|X,Vgv*)G20)9~,$45fSE|8Z6O}E_I^vBKQ:gD\"3`2AF)kit4hB~"
    "?szaq6tW}I/9ye%^tvZVA0U4[_`~L@v8DoRqVCQ{%|}Y`26LBqUqI)6/4FdB]6~Gdyacrg*^QDAIbj:WqyPr,7EHH4L&+r#LfqW+IY[D.ygqAJYMVCT|bbN?"
    "36zv|gzbNyGA4}\"11}Zl!AE~q:$~xq&X!VkG0|#(@J)aJI.}6rD\"u+g~L_QDPZaX#hsyA*%&7_`[QtqW0Q:vk+)h5FXFWj.|RQVtLu7v^Ww~Qq]X|L`H;X`J"
    "+_]EkxOX!mZ]mVU@|:HojtdeXiRfs:3Ma\"S&zAtyC&;+@V8~4,B*3Ye{8[|&L:hD#TDOh_/Dt\"f\"ft%tw?qCLxKXz/Ur{~<VsQk>@Tg)^KoC[a^Lo_KDJA8}"
    ":CqdGC;`,Cv}mi3_it=W8tZQzDzCkUuAUcv(XLjqZA9~sy:XM@ULft_~1K9~i|:Cd~l_:C_)9KCA!(T)/.6y$}%>|EfA~~!FQwJtj?}FwtW+7Kn`9~AVNVl?"
    "cD=)v}<bAAI?C`vwSS33](~v)XWBo\"W+k~zK?{~,Y@lSyn+Tqi$\"()}I0)7{)/#Aenu+/2)GoCF!2u7)sDck6>4Nb1EXu3G\"9(KjC_oLTV>Z+<d/^vNCSYK2"
    "cZY~$hMISVMVnb`7?zA}/&8sTEMfR`.nhdHLM:ftWV_gkA:a>j!gkkfuQV!M$K4W.57gkLA*n4!~b>]ReLb^hn/X,~$B^hiqOC|^swcaC3L.gtjXUX7^bG2}"
    "du_RwtsZI5%G!~7am?S?}p!W=>M`VqNB5hB=n_J*hzDBS_]X4M^Liq))J>x.+IK&|@>R.IQ@i,W(v9{ToAGr;[B5(G$K~1c~E?oinrW`[EnF5V_g@L6fKq$}"
    "jE#G]yUsQGi[XtEhYS~vdA^RW7&lvPUQ)|.}Y)>PESJtJ~v_gH%Q_@irYyP(%VD?%|xXnuKV{Fxa*>$FnLUDDvxQ{kbyE}P)GL~$}sVF:IRVt[(_i~FZ+>0?"
    "\"1o([&{EgA$0~^\"s*rQj.ChD8C#rZFqv7,j)A~,C}$VB%~7FA.;h*~V7:CUgC\"P(2iq_WqFxjgbF?~8B?Xp?B\",rSC2Kg~[syK!AvWC>dGTt,}\"L&\"^)7M@{"
    ")k#}WL3Fft]Xc4C\"QAw_csIYOLkM{y%)4d8ADq}l9Lm[RSLvu/YqlBoMyED\"ZC{_X_Ux~2wEAA[~4FfCg}@4@H]>!}.J%HBnFES&YL\"v3T$@d!ERBH;&O/e|"
    "`D6>_(N<C\"WW*v3r=JC\"X*4AnI8ZAY:{f|+;1K;~#pUx`J\"<F!hVbX$\"C\"uX~>~XfXz`N4K1>X^F]N9uQ)!~+|wVpi64>~v+@q(=/I{~&?u/qCx$+ATzsBqW"
    "#GCw0)R>\"~,F8)YVxLDt7,eXA\"^n=s$ME_V[=W,T6G&K4$tW8_dz$}JTx(j|4$VB/L~4_ssU_}6Li#M4VLFqzy1W!GhDPA.&ep#W.A;~#>/OhH:vn}j?SFlt"
    "L!Z+P.S6KxK&w`zD=po%J_<1GxSj+Gz[8uvg^(KCGD/VEWPD/w*hy4HL6]`>y/Ho[)e;5EV_k_VbgS0~C\"@bAt|~;MyKj\"v@i=x_+)f4>K~~cZ2(7QTn$`~S"
    "s.Dq5yy(&F`__vN&;>(_[YsWH?,yJt)tXF4F>%X4D]`xhSK>AA(y#}h>%[Zt>|6F;FMf3Wr/mxC&GCM{Rw.v2,&=I}vIggs_n_PA9~R\"OX7Rg\"7}!^\"CWu{L"
    "&_F|N{B\"e~BtbEU@Q`|yduSL~LCt)B_@Q?uD0ROCe_8y|.@J_Q7va82{_X&n?}})aL$H)Z^X\"]2|=)o%{L!v3u0*.=pLSV\"+fMv6vPH>!F<v9G:>xQ/s)v^L"
    "d!9FIA@K)kxzJ51<GIWx|sq@:4CXSv}LpF0,ZL~Fw_Z(d6F_gw+T489GKhj|MtbATEp[X)R?$Q>J[(r<Y}+>)_{uq,pP^D_0_~A[w/]k^sm?j_b1#rBhI>l|"
    "xBi+$J31|r6r|~@k7CJ@E`H4tu$*:9J1Nx=>Q?}1}}aj_E5fVZ|4?EMt|G03RF\"v))Utj^2_Bq$<EH{xRX4*t:n?X(|sPFBz&)PMo>Gt|$n=ZWcF2+7[,&bf"
    ">({|&FT7mu+XVLSk&vA*9_jk1(EV+mxNFpWujHWqd&DTgGqy:#IOZFU|2(|UsQ0]0Wa&q&SU;~;M2G/F(Fl%%BQ\"fX5LbG0CItr>STE|>0f_Gv;)Z?e~g~nS"
    "Ft~s9?F!T$r>+;1ZtW$>31B&k%RW*k)viX)>no=s658AItRSjLV_v}!>aF*A<s]K%^J\"SWi\"QJ[RcFm(ZNNL!CBq)tSX~Cw(%5eF,FHYVBA\"WtyygVv?etLc"
    "{>O/YI@`@h~FcF]$Mh{E=sAw.Acw;vOCeAXYj=vcoF\"X$}\"~9pjZ=C)_Btz#:HdF\"vZ||j%^}45}7YiGDwJV2i6G5FS&_J+_NGuuLMeAF/u?eF7v.o~=l_>s"
    "X+R>!~z_#TNVjh\"vb|DA~CnWpBQKV|&C_4~shqerY4@(m[\"$8}D.TD..g*M?2|/zDMTXs<c+@gFBWIZqgNHj=~QlVh`P!D`~@l^}ip!u?4VKL2s,i?oRX_B*"
    "WXj``nJVZB7F<{*ru?DAt(ltrV/Feuq?K?olVb#fz/FA?XE>U_Y1Ux5FrFL|cZw:Mv~wlB!<Pw;C|LB~+x9r[lw/SqO+oBVR4F:Fx%LI)6N!MV3RBDGxSL!_"
    ">\"+)!F+L1|*hI\"aEwk[P~FyvIAOw2WaX|L+4zCq~cM\"C2u|j9_x+`[4%#_,sgVhk9^\"F,}vw5)NDx(7M_Kj||{>T=VM5!2f~BcoiYZLjHB$n^U23K@fwZ}EA"
    "|E::.VbF9CvrB[UcG[t+}JX~G1Qc)JP(4og&O+SeBqrZU)4E?cqdSv{Wfy\"sh++=uAaixL+FR*(}(_l|Bt*Kg\"NZ[VA^)k\"X)}VQt~I*yWtP__[CE)_R?~o("
    "n?cFSqZ&j)7A~*=>&~RDJVc)4F7vcBdBo_EtJwLAVL6aGa#F8{ExyW@K9~Y&HAG\"n?YRo1#((AQDZwq>i~Atm(SX7L\"vCskU)>W_vVVx4FCw=Q,>c_V]TZm{"
    "v>`yJEG(;A~C<te\"~]Wv(Pb<k+?T2cusz|U4tG#GQu1f$aTARtL8(h4IZ+{9<Az6?E|sznl[#F/CeZ>>K?Fqrq~I*bStBwpu+QSq5Fx~~)s1N|fg3(sc>o/l"
    "/`b]pd:Lx[ftuWsg@Emi^XKrcM={fuS|!|Uw/,QhyQ?sOuu(&^,_2WAMo\"_C?Az|F+WXyc~^y#JC^F+>r,y3.>fC!WD7tKLIQYFq}~<Co(FpP|ZvJ*<Bp`/1"
    "N+@ASogd))GHbF0C}gk\"@C>>1W1n6`})>/uytq4*C~_q0,q@bGH`e+fLm_.yV/.J<QXLQ*}Z]Q)HAwJe>K`3Fb%2L\"ur*JN?0L\"tg=`y,yQVBts/fnjEEk4]"
    "!{pnM4]Lm|=(lB/Vh[o7GCnh^kY(PA+H&v;}(_vAg4QW|IDtc)<|(3@y#TH`51=X;jH?x|]9jkyK&{9_g&{RK`n(g=}RB\"XV7(vKVnOxRC_QbCIA_KcfBYK7"
    "L_:F6F>t/.\"s7P^GPLdC6awA~~*Z5f>>w|2B9*%_vA;>ZMit,[v4BM;sAtFV7L_|RY?vx?BwM%P5$^)qF/|gb~E\"YV_K+yIx=Jp?7C*un?.?YO9Za4{];~`~"
    "X)o`}s~wUJ4Md1vTjLMIZI9%[p3W#K0u*&1Q^3AKg3xJFGS~Z]uV.?<$`s,Hmk.~<sg>ssYv5WFn`kOxROlP$_U|zw8FcpY7m+k=;8!WV2ZAf+s~}F=slWH@"
    "ZLKI(C,v6GAzF/#$}MCt3rN+K?<1WWP<jAWAJ?M]y|>j!_nv!(kB8Gp1uuQh`RMpt++>CLfATXPRrvbEPvQRF7BwhB4E$q,Q)Afwa|Sj>W:C;C=VM{oFBtU~"
    "4~jn4r\"]F\"#}4%DAw}UV3h\"FJV().b|>),JC(~ty5$oB2SS\"=`4~LB#(Y)^K4y*z]LZLU|7)qNGSmhFE&q#^8{VZJ)iFfn&B/h%bjFA*cgE_II/S%h=VN2;s"
    "EM.>G4NBVJ8__[NA~FCt`0.}Z)\"I@}We/La4n:X4?KZvS|%W5Fn4;v!&`LRn%C7Y(AY+UJjBm0TE/V{Et\"2KVR/sRV$^5EmCfugs7Gwq~FBt&_;yK\">?hA9t"
    "n>=sKq;XC\"kZd~>/T|q|LAJ4HYP@0L=vI*(AB\";vUVb~)ne+6(_|QD2(%&BGJ1:v}saG4ytWFV]}W_9Cz3r>Q|2Wmuk?I|0+)**_^_7vY40RjtMc$AZfy&1*"
    "v)wq&v6W}E\"C1BI@^Kw1H+LAvsJVxBk\"3YEAD%V/1Kf^=zrZ$<6#pFgF+>t.\"4Zq)5.?_Hju\"#cTCMqd>CRB0|W+I*JQMvTSj#`W|LEq\"(K?8vE*e6e}|CnI"
    "#>~!*kb#2fINoI_!?]VYs^Q$+f[9fQt8f@mF.CBF*5o??{AVXL[~<C,r\"?k^+ks+{X2AwtU=<?beiI]~zSqy{z_=.bDnYt_tfFOD#rTqHB@jfu$<D_DA`ZjB"
    "rvQqGJrBA\"9B&C9~OiVB2SUEz[1BXv?QV_.$AV5R}EvdA$rW{_dB5*/>RLRAC_zDi0&rBXkRsZHAtyF%fgoWtC])rsTL4Fgtb+@P+[7a7(!?2_\"DHM,?yAZ."
    ",>EDCc_sJR&qHb*>*\"kxnY*`NJ0ZDMa!{x[vG,T!pcSyy]{/#ts/ag6Fygoas(B>6CDxQ~Lc<tk&]q@:KF`DJAJyN+y?XALEJAD|Vmm?X@Tq)~iXJ`6/8x~2"
    "SLqyr&y6_(7vZ&\"K)_51NVsw4/SD^CXX:AYV?XCGtyHA3FinuuHA7CQAeGznsBr?8FQAr?o`pFBt)AQD<X8M;``h_v<~udTtB3)AayNE+>b\"=)u?(A`~uO_y"
    "\"F78Y@*_9s<U,}mVl|Mx,LuQA$7s0*^K06!}?)>FkAheIGx[Pq5+hGi>Lq2CB`\"s*W,XfX([8C:>(>Sw|D]BWELydm|LTFftLxp54~\"I+oY|4L:F:,dZoO9m"
    "51{|w=J41W]&@XZL^Uo#@QiwHAuQODmuK&<?J7bBFJ`LqC@bRhm\"Y&$V#rY11uQ<xK?{>$X)lAHAE_a]wrJ&=>4F$rxAg|FG5Ma\"WA&Am+f(].Rt+T$MK>{n"
    "{U)h`Fgt\"~)Jx/csy,a4n_d1Q&Q)rWu~jEU)S?Xfb&2?6L/fk/I@JC2_F7~KEGQJ4/_5>Jity`8A{0+rm6!_xA<Q._G`5CG>zKZF~`wwE>GIUB+>5:1kSId~"
    ")_Nzp&VqZLyn&v8<Q?%qXVgg4E^h@ouWdGyI<sZV[F/F2WVtzLT=6X~~CH(!p@iLB^Rw>oZB1L7yzu[~oVRAttaLQD*JJV??Wyl7*h)UsF22/Og)#KlEDCVF"
    "r~!ugM_EAtHt{q=(!^8MZCd~!vx}_N@K8fx)n@N)CAR@n_@~jBW]q`p[B*DAGp6B_@eF$Q%l3?~EG_2(qK^Rbl^UFAA\".$R2p_rAG>9GR?Fp!WV/k_5}o~<~"
    "ryaS`s8_ws5CBMkMAt?C^<K?[|uu&T%`R|E{D?!FyNLE!C`FAtp#hBuQZ[nBI4Y\"_JcZ8}%NkZgI]KxDq|ajrWsy=p}@J.&12(bsNX?t.([>t_=ksBGh$GRn"
    "CAF=xsZVaZf~\"Cu+[>K`KF/C};TK!ttWzY{KR\"{L0Q]nJ&CLE_$|{%HM=JwDTxW+,>4FdZngQ)Bt=%8ArF1W|smG;v>G5hxLSt;X~JuEFtz8AM`L71Q*UVg_"
    "UhYVbLTR?_#W#oAH4F_)N&j_^>s_hu>/Pt5S9[yLDAfLS)7\"uK5L<FBV&h(BwnEElZTRRz1B2>VK8CN|okqiV|W+F5^K.~]ro=!^qFdZd~z:y|#Tn?DG51yS"
    "m%l_:vUEv(DG7CmuA[#G0nYtb@</\"yZq?LCAIA3F#1+$X)IFH[!([Ll?ov6Cyi/_kn2ZM)8L%tQA}Ec]0uhBjnQA`~i~&[[C=J]KBz5CTL4El|V~|Lk_DuD|"
    "HX+_%|<U$Mo{f~^)VBEA2T&>XQ@t!I4MfVO]<k^XN^fw.`9hKa<mQtL7\"9EuzWM<p_eq^&Ah)_#{>Tc%{R4viFLjT~8CuB/$+`Z}.$yugBzknZbLdFAD/}_="
    "@?pr~$AYI\"UBTLGcy|uu@)6F#wt/(fdz1[WAj`:s4$KCS{Vv|9HY&>OwdB/J$GknHAZFB\"Qt:jH`.ydZ|L>A_)<V:.y_pyGOk_9p~.V>+^/v5&^jfF1_iybg"
    "G>U|dud6fAtBu*uBxqtu)V$>n?tu(A\"C;X#fxRjt3WL)Q`r1y:~=v@wAU@_LII])D$7A9J^XTFy|!RI@{F+s8_=)}~K1E~yinBY1)BzfQ.qF#~#M*`E_S&xW"
    "\"}cv,(Tj@Dyq}.Q>!Lui_,av>_zLVBHLN/Ho?TsgH?tskB,XrP[_5vtip&Rt4}UN[Li\"3@,g;C$T<HDHDD)sq52/Il|s!W\"]P\"zYH`O{b&J[,>7<d+PLgEjq"
    "7[UV(F7~bE[V]K${IYxW=|rvLEE@)~[{`~M5{F@{CV|LaGj|8(I5zK%A%OiM)Ok|B\"#_xq^v6?\"^r]?}a@j_l|iCWN}]:IPw*tK@[|o|cZwKI]+Gx[$?G`vt"
    "M5)>AwG}vgp`ht+r8MYG~b=(QAIO&C?)[E0kU||)3})>3}5Kp/Rq5,9ucFW|yT~~0RBqRVxI/?pLFxgZE\"7yaLZFq41s1NiMFt:v!K._/C.}/h4E|_|rz(b+"
    "EDI*uWZL71kq3(,g$~9(Z>0/bCC%<V2{n[zC#TR?BwZ\"<>6FvrMt..fq?~K|!=LO~Al]{FSn9BBM7=aIn/mK%^5sWx!3..h\"iCt[6sTq%1>?5Lj&,Xp`AqsB"
    "n?ZGewjx:j>I/CY*DYG\"<)7M.?#\"<tM`x3dE9}BM(1jk2KAgE!NZSCA\"1}~Ub?XWRDNs63<EJE{.Mhz/JLzZ%v;K(_2II@D=fG=XK>R{/~[FzD,`on4+)qtX"
    "}C/s)VwEIf)%JCtVjqmr^TQQC|c$DK>Q_kbYzgqQFokc%hP{9CXuTQ7~3vzX~UJ^JG(,QJX*2pL&cl?&g~M{AMM!NzYorDM@YI%WHW=Jw{P*N:&~SwW+~qgL"
    "K4rkP.bM:v\")85~KJDZt.5+Bpfa$0M[(jk^X6Wf^F_]%C5]L/\"EtzD&h5a}U|F?qy}QVR?8vO!0*8LwqPm0N+Q^9<BwAL.S\"?PdtO+!(2R^35T.4WL<P.zFm"
    "8Sn|I\"dBGtTy?Q%LpLbuOLH`vn&w6KBFZ]Z:pLaG*+tEz3]Q!wm/IY~~`Eh1wUTX%z>+hGo_6COwk~|KUqF%j?wRwtD&I4@F7Cza\"e`(ZIjqJh%~XqCv#MOK"
    ".y<sb?j_Wk3W3fg\"Zg~J8AuB?CUKcw4rMVwQ`EzBCvJ?A\"l*Uh*~ADfZ`~pQ\"4JVG&n>7O:$N>s/eGtZsw=_N?&CQ)cF|v=(X3`}SqyS2h_E=C4F?4hAIA|L"
    "?{HAN?d_GBqWTQ~Cmu?LJB@~uraCYGtF]XSCiG=yf(E4PW4F6v4*TW%|TSSL!_Aw6vw}w?m7DciW&AFZQ)*_3yVZ[JiA?rp*I`a|2W\"Lu@At\"FjLyK;C`~^s"
    "|Fgt{DXL=J;v_)8AayS&V&5G.yq:kNYG,C\"vhW7~;v|}wW8}}>@r*VIicv$;zrVF/yZVn(H_$qjUAAzqVB3KwKIi_sY)AHz_Q*Cr{~X4(:i4D\"2F+qQWno+T"
    "Bw}DKfBA..HlF%T@9Gjn]XVidFq}9TCC3=jUN~sM.)|ua|2>(}bwF{*J]EDrnuqUH?^nfZF5t_(tX(FK1FU|F!oB!_kF.(:u%H*Ie(}~DAVZBtI`k|<Xs)J_"
    "=vqaGCA\"w|8,$5J\"@}vi=A)VuuZW^_S#!3DGEzUxr38~!w{5Mt])\"F~A$?z8FzGDw.@?v~{T1*!R@{IVA),>}C)X6[OR8QYVLXA\"7yWxc_cL=p>B8}tWhD6`"
    "eLMGry8:Xsv:s?1+%av/m!kxS>i?yKxSz?ZG]6P+s%!~,|`T$MYSAt+r_LWY?s.}p=hAW+E5]/z^bE_L/B}bJ\"&`spL&HMN?qFPAX?q1DVng_/`|~C;XM?P~"
    "4}IZI\"{.5W|LNDo@<JF/az[`4Y`~8~CUZNk.!QzEjDwW+1#WK>e`cDWmyW$_oqc#9h|~S_Bt@tD\"_snUQSs7{Tde#B^@X=R&5~!yJY;$R:{_%s>X?L(|Mx{i"
    "v.DAV~2XgDPWT41K9y7[FV6_V4#A%nAtV+fE1~DDG+TEwWh\"_IyKDq{DkgtL%\":YwEol^%nP9~9yANi]7EOGNZ6f_c>FBA=R#k#TdBm_5C4r})FGl1vBXXF?"
    "*k#}4MsQ(K5$l]ZM$|pFZBc!5IN+s~_Lx$N{tiZFY4?T{u)^L%a\"bE4Fd/*>}DADi#y6M/dDjBj(aFyn$rF5i`RnBX9h9Fm15D\"+K`)zbZpp$^_D]CUU|Q;F"
    "ZAT~oi`X4Y6}XIx(Gfx/]6k_i?g^ZFNrJEGG)n0,_N6E\">bql6z)tp*++Tc_TaxaMj?L7yVmWi7F_3?`iuE`izh8Gj.&Uq1W:9KV`y%az}DI(_%,eilVC\"U)"
    "hAq$CLZ^k|C\"UKnC6|vD;X*0e+oBn/Aw&]J>MhZ1zE@MAGZILc2MIcGh>b~Cr(G\"YA/FZ&f)!~oOOZNhVF3_K*hh=?;vgF:&w(+[TE$A_[Lxj|5:S_S\"I.Et"
    "}w@JK>7Cy}D@VESAEA\"cPABBeDXoy?JG:vrv,A<G|}L*w?my6ytKK:PAbX+CLL$({|.`N1m+UN*PfcV/yKn=AA;|m,AqK|ui//dDa&#<$nBEGY]jVXjI=WG7"
    "EBwkLAFB[^(v(M#^VOrC;|CFND6C|UT`+FWBr?t_inZ&e6x@m1W/,L0Q>t}wCF=PT|2(#M_E2|Uu}@6R{0b+<4WW1nDx_g&MdwA7U@9\"Bc>&~Q<s(`9m5Lp4"
    "/C`~!LnOpYW2a4K}&o/@c^s3nWSXdGssS&B\"@J~F$.+>tQW4.}RCA_iz.$S,h>#m&vDA~p>((M!_2}:9JO*~OAfq1XoO@F33B\"NygWA46]~biS&Wc\"%:ptl?"
    "&_?)t6DGRtH+3YLbO|U|=2/bKLJVq;_Pz_^s;Tr/6vrahBC\"YDB5w:,1)sE$)V@mOmh=A~aF@rbXG\"C&#$EBa1ur#}%\"7X_)S?/4Z&EA2[t(%hZFx_Q*mXzL"
    ":C#TDvIA)5Bh_EBq]C8Md~fA%ho\"])@V]Ku|Y1FC~Fb?cRV~IV4_kBR`WJ+qxdlt5)&1\"N&f8>/I\"&U4{Jw7q|h]A^U\"W&QRL4Wu8tUL8{rP5A~eGp{+[)<~"
    "?}U>3KDq0Bh=](FqAA(~gnX*bsVLd1sxw1}^sC0WOV*~@5lB,H5~En^Xg)B\"Eq]}AVWK$+TpXL5F{h_U=>a~2vM+y(QW>{5s<)%_VtX}#}tWMIsum+HUZ?_u"
    ",TUWB\"^Cs={EIv`)^LfN2FEx\"K$&,hZ00j/ONf!J6DgoA1,8ffVQJzT,Me<ya.Z0xY@O6cRg.mfoX0?5wcUDp<Ua6db9b`X\"coiHn7=1zWP%S,cep1wGZ04j"
    "GP*hRo.5bo2Hw7wYUDm<k,<e<wb`Sm>B>OWd]1Lt%?!0v5wcW9Lz$w`d%%T=`$!jpOvYG5r=PMZ0~762(mvOTa^d[#{k`$IN#Ofe76r=Id)058/t[c?n8DGe"
    "6+|%[Mk*(O:hkvr={A%DGeE2x(W\"boj0}5ddkzrA9@/P{e~oKAQ%8D{ea!RD`$IN?Ofe~o.mPMx0,8_Yz2.ZR,ndG4T=Z0HN,ONeJAgAAA2erg)i[H,$66i~"
    "BXOx(fd[Rj{Tw*uQLnq*MtWMcHb$T?[cOzGE0I>>6}:[B)eQDfXe:T*?|Hx#Wm5}k!LEAho<3xu4t*7P]oXu2PCuKI`9lB/HO\"ACgQmq_3wBYMEIv$`%2d[n"
    "CESg??(t{Tx*%Q]kPm[>B?cI!%b,cLw\"`WAPjf<vQOCuPIA%v;2d7/sDGhX;)6:kCCEQVnAzROWMRIy%=/%<jRNxWgP^G_{$xYFEAo,!)F.?SIMQe>:h*LME"
    "+fX(!<LRHvkQ_m@z(}rX{H0%FpF0)L{wqh~[PJLO1YQQ|n.LDYtXSIC(,(jG]n~wHh$_[Dkc_uYQVl*0m#Z5@HG!x*6ROGNxjf{>bF;kyYxQ^p9s#k,idCf%"
    ";?AfjREEBhM@V3#FB)pQWeobxuFuYIp&`[43+4JE2I[=)\"H<s40HN&d/!*[nNxJh$}|xX\"YM~Bz%V5onXiJxJh::U:Qf0Y]D!oAT{v(i|B.##7SJD)MELg~5"
    "fzZ\".?CID#3^Y|vtLx8g\">MiC?v*zE0mBw^oEB}Hv&s_>Vk!Sx9fi>w2AioY=PLr10:Ta5gI^!C&n6MG6tFGa]3xzZGvWQAoY7}JCuPID%&]tIl4Jx$gx[W)"
    ")WCC|Pssne=aaM#HAAgAAAUZppK_p1$r\"2lG#p;vL)dMn_l+4h(^#~X}R&Q/]^9_4*p_bfm+zK).:/c(?Mi_,CuW6A3Fl/K<dGl45}a+e~QA~~P:DwT&hlwc"
    "\"L7P4PF?By$}m?P.#sJ\"]PWqBYmXp/[qdBn(p<JLNZ;Xi_OwZ*FO[;by95R2YF(nY&IM2(\"sR|Mllc*Kcq})=>$Di$)5t>2L3Zz,O`ynD*[em_P$)X&>,.St"
    "[a4*nA)+@sJ?9yPxI<~FR_}.oBi_PqDy/od^&|4q`ZlH}CxqD3aFn_*}.*2X1nw$/&EP][\"Xm?p_)ECfNV5~Cr!r08q/s1?GJ)a;@x^U+>>(G|Sq4Aj|G%aL"
    "@K!~Ptf47)lxJ*\"6(_1h+Gg@VQ|q3}v3y_EOm+zA>6A\"QJVR7vH&}ZJX31MBwA}CW+WXVL,Ia|VVwQ?{z)u?q_%{zx\"LiGEk&,IhfF.1IAk_QwdBlN:_T|h}"
    "d~>(B\"4uxik_gtAtg47Fk|.(~~\"9MwO;h~q`oF~w+xzF=v&lT4BMNGL=i|eR1I;))A#{$rnLl^NzJ#3lr?kqJwpiYFZ1|W(7o_0_S&#*{Fa`v+Sv1Kw\"0AZ<"
    "qk1[cF>_KVZ6b]TLVBD7YLk|:XoBCMRGeW/VbF/kSt8*&^qyTaqWBB=sBD:,?KfT.}@JLB;s|`+75}@KIYE}B\"/4{b9hpVA{8+S|)`0_GYAheA6$WLCA,(0("
    "1KtDpFyK@K64)Bq?YLIIza4AjD\"9LLi\"a|=>/_IFpS|Xb~izsu1a6\"TE]J//zENZhB3FG\"qKNBUn`)T)h_|y<sd~:b8CJ\"s?Az=r~~#_h|UOJhVL;sSV3i0K"
    "OtY$y(y`4FMEf;M/31mX4)j\":CsYU::s+Tw(.`Sz&)QV,b6CGZbX8GDqry8MYEHLWY))4R<F`2#KZ~ADU|o%[K]_7~[&tWht{{)AR|]$eL2Kh|_)B5i~uC7_"
    "v(zW@_l+GL]KoIJV&>czV4XY$A+|n&M@?KjqvrZ~lAVZNVQWzn&C5KOWoIX+kBq_x|Sq}@|ELFc|?)wPi|mWbLVRit6v_J|LE\"Y~{L]|f(XL6GzqJVf|(O/y"
    "4ru?hGmCmu<V3FTtXu`J%=+[])3?_LQtkBx*>?51?.H?!FJo!V%<uP:s=W<h5BLv2_R2^XJ4iqSV(,&~N!I$y(bC_%$A4y}owM[`oFAY*5+VGoWB{Nf^Otlx"
    "AAUqpSw}rA;ChBC\"{TrKQQcD|o5Mf~S|<B_@8FcFQ*^)y(lqmB8YeG_[h&PLhz>~L0j)t:Cq]XpBo\"A\"+&GVJlV~QVTL51l+5K5RCq_~V~o?vqvroIi\"ME:>"
    "l_ZytBmWhBW\"uW;VqIWZJtG`9C\"X#}.JeD&a$}A^~Cou/O1FVhkW}@eH~1L|.J=?2|(aEc._l4^UuiL?B\"*TIAC\"BK0Wpv6,aLm_gAu%H?j|s+$Uo_SDtuX@"
    "+~EO9+ks!>FOK*;txFP2oS^vq@5senNOx(rtX*Uh||DwYtB*iB=w?b4}3FH`fSm6d~e|jE_@j_9~4W/V@Kf|[}ECY!R\"R>rQT|e+dBcG6yLt)J;/R}YV!!cL"
    "gwn(>&vJi|hq>Nv(Dq6asg>Wy|{Gh@#mBt=~_~iF2yXVn(_b{[?ej?1K:v|TiNbFu~rn/J)_bFiFVsx?8s<~ejw:(KvrS]:`vtzFr(=W|]bE1th~VO!WFh1L"
    ">m,$PMC\"L+MVuQn17,3Mr`c<zPc)xQ)[v}])PWQqCtx*h_ULK&=>4LFny|HAnr]yj?}Lz3AY~N,`ny!(3AE_rX:>?cE|AV;jh~UAn?K`mqAA6M@sLHqu$^IL"
    "S&AAyEdZUL|FfAv(5YnCUEwY,?QD^sSL/VCw_Ur?BALENA!vjx\"#Q/AtAA(~:rYgPLp/rv*W8M\"A<vP(qDK>8CluAA9I>Tb@J_hGfubLi_itcZ_JW:L1cxo~"
    "@LSnqF5[yLRAfuvW4vhqCK?/BtjVyia\"*)#<K?BDaq!Wr`}L/$o=mhMJe%=>,>20^`NC&_k\"9h=/StHtkgXFdD(,yPfFA5P!DMcGgD~*+(\"F:s&yTLL?0k,}"
    "[)7KPwx1$M,`N~EREM&_;FDExW+}=C3rmBgMchQ+$<KJ}C>o/V1F=6V+3k)>8wTHm~S/zO:,SL3~[s8B=@jAvr=>PW(|O+YVTX8v^sMA&|bZW&B_TtUWR54L"
    ";L9r/V~~XIDt+CyEfDr_{qIC5C,o=h4:={uWsUNX\"vgV*`s?ZvZ\"?V4yjxH#B\".C@.s~3Q]K?}}=Hg};+}IMo_]sh&8M$Gg$4T(LFHavT)_44F@[XV0}vEw~"
    "fu[]6^t~0+Z2|!iqlB9[hGi1!Wn?\"~Swo}gBk_qy_s)[m~/i9sjLZLiq$}[=O/:I_)6r:_Zv%C~~4L__5yJhgG:vJti?5EPwhd|?n>DwkZ4ADnaqK>!zc?tr"
    "cg8\"IVXvVGi|lr4MO)AD/ot%3QEB|bYAIF&};*[K.FyS,@M?q1U#oUAGnCVZ:Xb~Wy+Q*tXLn|tu5KqQk_Z\"_}G`SV2C0Q|Fz,FhW)qD3$9Ka\"@{xu3FpIeW"
    "0YM>}.j#r?URkq,z=ep/!1$:A1,`dzfAn^Z5jHZi{2i_dB#7kB~u)<rU]mv9K*|)2/}7lx>LZrsfeWMX+_Pt/S>j`Jgf8M4M#^|v]XyK#~RAIAK4o}(}9Ml1"
    "`%|@dA?TiX#~cvbZ|j7~6yMcLCw:7MGxWXNR\"Fd+857GtsuBu{A[,CnYtY@Q6I2j45;OqiKV:Xo_(hItps7TTtZVsMaE!{dZlUXFpvBq3!`X~n4bJ]AB;KA."
    "Ts/DEw9rf~hAu(Dw^R<vMErLGoNsU#~xr`3FYr]>VE=~Nm,t1M7vX}nf7^?E]:A}qPZA8?FAl|T$#&?m1Bs[ZM2IRY$A$qAAr/Aq$T0A:C7X}gDBQwAYh6DG"
    "lqi:tWbFPz5yZN/_~yWZ1W@E8p}.|sXFry.C/5#GXL!Wg4J\"x*Yxr_0nsWn?,&EtMY)*K`ytku<@5MkI[}ZXR*Er^3`>RQn4p#}sV?,E5d!KAGz|IY6mL?Ot"
    "A^gNP/ss@Q?&(M%N[vWvoHG?d^E@KJk_i|kB<cssTFXjA>Bz3Tub2EmL:}V>RK&4DE%*dA|Dh&E^NGM0(Y>QOtEpf4j\"DHZ={(Jydx=~h_F\"2DO:Uk2+.h2E"
    "R|q|LA;CN+=Jl`p1+rsY1L.C=~^LOWQtgg*>TFqLcBPLiGssa&:A[kK\"KVhqdEyXwQ<yVxfLTc=s]Y3YA>HIz#VCNj0_?v\"Xf~%Db|OJHNWhMZk38R)[*)(t"
    "2FMf]$>M5Rcf.F<V:EcIc&$tM|l7m;;ZVR~4vuPjkQoveW+>I5BOeuj)3QIvMEsBFnyDV+rI8GuD%rnbEM!q^%.Y,P<CwIx[74(|w$L@CBUQ,T_L7sit{T@1"
    "p>0ldx7YFBeJ.rXX5~>6OBRtBSQ~$.$tIB7y8#N>XR_qKt^j!GII*~ggN?f6u>6,*iM`P*1B.@ht6,0psW={YYCa^DS1luvKr(k|Jw1WV(3C%V45m`Z4IwQV"
    ";/k_^Xk~b\"uW*h9L@s%${L=VxwBVf?<PS\"6rhGn46CBWDBl7auLMl`9sS*Y~m/,CB\"jB_KFn_vE57E+93TRJHnY4=s:OO!Woi}csu>|xY}6^XEgtVp9CSQB\""
    "hdo=jA6X]Oe\"nV|LCmgw@}\"LtBsf=Za@qH\"CV+w?j_YI],J*n_rv8BLA_0yngsu)\"C`W]XZRy|,(j)vK]_a/=&vE@A:?B\"NIa&&H7~%nStqu%~ciqCF&FMD_"
    "`T+&|Q>~et;}h_<~OBAY6EQDY&u?*A#T`/+(AwrB4YH`T|>(;|L>+IXr2+`}>{tZB5l_N?lWwws`n7)sCO%hO2[X`~}LStpy.jl\"QcvM:{%NKf|L@KK1@YIs"
    "iA*(UA+[ng+q/?`IQVu(VKcv2+[2g_4C5r$Ahtb|;A?NrvU|LWNG8,X)I?L<^v*Vo_DtPV;LP/?AwI)G)|at^)]/6C_vsw8Gg0Hw,|,KAz~X(|J_&t>T)YcU"
    "Z`kB&C:Q_|&XK5}F]w0+)V_W#A.tSEB|+Ty?5EPt&C*&p.+1WWn6$Lxp{)3AAzYqLA1yA&2(+bH?4qwg8G]|?$?A_0%C2C<_f\"*m^/Z`\"w&&M?E79n9i5JoF"
    ",TMM8GhQQ}^LQ_mFrg3^4RJFHYYJyK@A`VA\"/~1W2Kc\">T[ViA3(oBg_0qX&m|r>X\"h~@J`F_%DM[RsAn(&_!suB;X7X%~B&NCI_FG_)ROyKjtsB=J|FKlS&"
    "<hL`l\"`~{FND_sIA~F_)RVm\"qETLNPSn\"XT@q`Cw.oS>ezn>aSHT1L;Cg}ti#AQ&/J|~GI#rTXe\"~C{V2MrF[C|LVRQ|vTfX>}w|`)@tyQLIDVh=WD`[QAa_"
    "2I))_)7G;siZ2&_L?shS~B/?RD2(~>ZM7F}Gmi|G:1<_u?3@.CR44}%_oIm_XsF`CA@)N@!yCAbLG4}}<hx9iHUnvg~Fevwr0MTd4yfu%J]QNz]WeiyQb`wF"
    "@4_RV1o1=LzRgtmF~~6LF_Jq!K,bD|h(Q)#^N/KYh%{LW`\"&kgbFW_Dqu6c~dy2rzkY\"))}~VWC2rnKLpPD_(n~B>oEoK&pKtBvANV8}J]MxOA`hkBib#L9T"
    "Tp~>aF(KmWJVeg.yoFP>9^4B}`RhFBLFS\"Z)XF[yz30Rzn|A{W5]i&Eh//4FMcZuyL&N]TWlK_iADHc_ke^vC>|)iq}ADB]HnWVBgGDwuBbuy/]kiqg~$L1["
    "WLbLiG,_ork=F_B|/a[JCGQqp}HMsW:CT&iW_ERw5:^)KBV_[93(,>A\".ChuYGlt7,LLwE1_YV~BwQOwW+?@*_,kYV.ML`tFjy,$QQ(nd%6X6\",}#vt?5[\"U"
    "d~n_2LkBxI1F<BQ+{vs86I>%w8I(KItu+Amye+a,@~A|LS}Un`sspc@saGR3R:;X6!<Dp}x*RcNyuuoUeFgqFutt%=7v9Z9J@KCNCZSXH?:vRae|obL<$r&j"
    "U)(_[XmKYM/vH^dNhH/y:C,j,`jn\"sFtvLx$8x`>G`a1@`xhk>s~7a~~|Gz_#$9W]K%CzC/VgGPt7v8*}F2[]Xq?xQQw*v^g5F)_xq=tfLLz+(\"j8_/CT|<V"
    "/HrFtBBh*JAG};N&EBZIx,3M%~\"C=a2WBBi\"aL`F2L9);V*`Cr^a)AV|w}|LL?oL(S>>_Ed1i&PX/?8~a&.MXLfpQ&%tU/ODOxB7ZEy{X+A\"hE%n>pY~m~^@"
    "\"v0i&GaVNx*CPRRdane]d_|0C&x(dgtygS9&H>jU*eP)i=(0;v(kbFnoU|[2DA\"C@.*_<C$rlew))FuZ*VH~@t%,7(M?dt&])t/aPwQV\"&4G*_Tq.)[Ep>i:"
    "LXlBwD_)@luQ!qIAO@,C@W}J&_F_AD|@[yM]cZ3^n_8Cks,3l_XF5S4M(Vq~gFv?]JIIgVos#^xt_vU=`FgA%t/>9m<v@[.b|_`%h~94]ho}f8k`=p)yKOIQ"
    "3FiVz3XK)Q\")i3lIw|zuwY5LH4G(>vI&#9O}`NQ:hq(v@A@N8,J{~MXLIAeA2rFN(H=p4C(r<.3If+pB*Psv~Vh%*A)agk>}<yfV?r7^q1/a>>cFdDBtLLG_"
    "X7k#q?l`BzYVUX~F!6<WHAJG2WQJ_KH4nV5*r.AtJq0hjh`_&,6(V}MC{vDvSQ1[2W2WM.Nf),xWRLtyC0d~4FZ13W\"qMWmyVW2K+_y6AY`NI@wDMcoBtL7v"
    "y_s1=`XF8vg~0/s1rv/~7yQqB*?j9~`07aeLcAvTTX~[BAZNvPnFIA|F<v8WmKk_Bqb&.hAAmuWL+>h_)~^Ag|gV[Jv?IIM~tWbFT|3rl~A~:C=)LA8y}vG>"
    "~E#k}wm?#\"*rbLN?5{|T;X?L^n^)q6!F+hRVh=V{QD%X|Xu.{t[aogh^jtq|<Mj?X4v|#A)\"@4YFF\"WXjFeAG&u[]~,r)MwE*[+~MhV~5F2+Q22K.C#WCC7~"
    "Cq4}y3*AStti$\"?}oB#~N\"yK9FPAaLXLZIW+HABA4}*\"+T5}}F0_6C?^o`W4}9/hRRD|QAm\"cZ^LFB?{mr8Akn3TuW~F/s_Xmuo`0kdZ7v_XuzMZw%A^mk]C"
    "MESKaF_~j?wbZI9aWj6}dpS:DA?{:v{LgM2|wqiND1W4IYNJ3L5yH+hi%^>jBAt?R\"4YI{lI}T2(RWzq8rDA?~0+`2T:L?3$lNs@{3,(cZW@~~x,u(+AaVcs"
    "#F4yx:>j=h1kFAeG=y(,vUbG%3}}@@z.Ctc_Gr._~IXY1Kx?0k$(L)d\"0CXs/``07}4[:/9c+ry{Pc}CItCXqW0nja})B)LyzvvP0EAtbdS7mB2qk|_~bGQ|"
    "Jc,H=(MCX1;)u?^0G&aLGQU_j:]LzF}IJ&\"u/`!vcW33E\"0/}CB^^k/wSLJ`fAq*i`4yiyW|udvq`BRh(>\"`Tx^)uV3H\",>>!L[=Fpb?=i0|IV+>r.bzR&4["
    "k?.v.}1Kb~=~v}Q@CB.F%a/huCBzgADB4vySzk8FqC:Xbs:{u\"1t0MCt/}&r6F_=M/<V4F}>Q\"Ei>s@QuK2GEtNGkZ0F`4_)IhpPmFg}93r={0d+EOS{,_cc"
    "e+v!\"N$(4MRB/k22AM&Ao(VV.>y|\")u?]Jzty_)h`F@~AV1MAG?~:,iL8}iq0BIQscwQw}8p=?K4bZACl=5F\".yuA_\"IVR6*H>|n.QN9$_BAVV3L8yv}*>`K"
    "/~dZ#T~Rt5wrY~^ETAwM`X!s(B>>\"~;yx}MVgG1t1+0YJBCw\"l|L3R&kL&<JH`HL+rngL/Ho:v/O!FKtcBv?1:MI:C+C*=2ooBCKC\"ytBMBH#I&z<VXLm1#("
    "Ul!_3p319a1~ol,z$>&^=5ssg*rW)q=T~Kg.jJ:U8A%n9~Dfx(Q\"$MB~6C$Tm=W/aFCsQ4o&/sT|LM{K|F6aNW;(Rq0,MVk_~CXwDAqCA!=>#\"+(w[eN=C=~"
    "M5L@tpzPJVT/F]m+dN@&Q<aS&KpV_HL,UZSKZ14|VC._YI^C{q#BJI#ri~,_*[mWRJ`FsCt+S+K&9AAkrB8CI*mU]EPJH.YgLBIirypZwK4N3V47~swt.1:;"
    "?_o[6&1>~LG][,WbEM1kPtC(1eu`jcsY[b`uu#s3JB@FAYQjq_Yo[Q1*xKz6eWPXzLGhanRJ#~cF(_GCF~jqjX)&cLW|~v2&WS#?#W>~(_Ky|{M>7_eF&y]J"
    "C\"e(U)M?xGC*7}rW?ph}2i!\"gY.QnbUqNBq?V@xt[$.s)=|v^s(X3^;sP!OjuR%QOZR4FACt&O7F1KB*ItrPdDotzD9Lh_qSLju`KM;v@)J.BtwS|j)\"E~rA"
    "lIcHH2o:|s]`|~~Klb:}_VdFAwvdmBi\"`T]>x>CDMcx%I`Ur/yiWNbvwNEV@%_/v{D7fyKwKVZOj3LQDn+DwaLJ40B`VlGk|z_2A1kR&#(9_%q<)w~(G=su+"
    "/>7G4yHWZJ24`6%}h~%\"raTjg^2B5vb)MH.4/,X@$~St>ogZTLU_R\"VEH]l)dNg_>s.o`h%~xqNcUEmA(up6!~G4@~l6s/j_,o|)WXOtuWQVZF1[tBdNFhWL"
    "|b4}/Q)[Atp[TQIL.${|F\"GAAGkt^X#M3F4F0aY~iH&tx,<hw.4v,Tg)h.^nSyM}]3/FkWI@xQtCjqYZxR6m_W!3i\"1WDf^9{I(vu[PWkn\"XjlYSSqP*V>e^"
    "i_cZNV9~q|Cq%J_Ec]4,]LP(\"C)x0Uk_m4=%]&TMDq9W(Mr/9~!(zM4Kin%F2KgA!r+X3~#v2Z=VC\"HY1*Pc][e(hNeAi(s@dGCwoVdNOW$tk+.tp./C}`2K"
    "E_R\"AAi|2(KjcASq6K7F/yWu((hA<XwMiM0NzX`~$~rA%hPXz>vWr?R!V|+TAMb~8C/yLE`Fet_)A*e~m_,bzAoF_)FOc]#q3(.55M&$<)+}<VMwmu4*B\"gt"
    "|.7MTFjqWuwM{G1ITyqK6y:yjZiKYS]q3}zYCM7IBw3It/CtlB[Ji>mk=stBmK0_lu7vJ`tf`;|}94BM*U?&4WWy%y\"L@irvXY4Y$]ft((~&+?AnNqL37A&,"
    "*aQQQtR&z35AT0MV|zUtg$;2*A8Zd&N?V7sZ{?AH7{>(X@r/UnyEOej`etSHIoL`0kd+m3=`8/P&*IPX3yeuD3xP*|%Cks>Q&kK//&YL0kJD^Uw>+h8Bk<C^"
    "P~9TJ>M/.?u+0idFW!yv7,EG>C>ru(`Ks[Ss@Z=?zK/})qwc>s3rt=;>p>]*nPwQ3Ck#HX9=Z]9W(@W53FIqIMT`;v)X(r#F;Cvrbs4EvtcELj9,V14(YhDB"
    ".`M!i(mi/1]az}C^#vQAe\">G%JU@/1IAg_R_&vp%o?_>)$0[|4x3/}wT[LLl]X|XXF@9sW~=}Fkn<~.@H_\"CK&dBCA~Tj3:?&HA*[OTLyq#rc~6~aIu(73*A"
    "C*uK:Wn12W,XiG`tA\"23[/^6o@u?K?$AMM4GCwZV>hOWsyL*Ff\"E\"v!W,j~FwT$Wd6M?u{0(qj&_Dz>)\"Tinj\"|)VKdCb&;X,>Qi1ZJ&;h?^?}})@(h|a<CD"
    "oA~Vd%$R(nQ*vM{{m_bu{6E_^h;X|+`!Y1o#TOxLsz2T4U,V47uTavdM#1p`YNpRBw1Zr(7EeD}CHCIQJ]oVj3?LF_?bcq]VLInuo?*_GqkTo.=_61XucBp/"
    "qCm+TADARhjH$t{}L}kN]Nw+SX+_Z1sZ$}ZMuvVxFtzF&\"]>;>}yzy}JrPWlX+$tyBF1Qteij\"Wu&fRR^AmFzPV<1u2AbDk:?;mHz_f}I*zQX1g+(<RWZy$V"
    "b;JCkts+/CQcyKy=%{_F_3S_ee|Fs~^\"VKoyr#HhUMx|v(sB$ALHUsob@tfy7X$G6/F(11u`Rkc_rL;PFna$_};QdTT&XL2K*OyFe2FG>{<B_4oG|13BrW~F"
    "dw#}BiV9gFGmEM2EUh]*{>g~VK,z3}$LFN%&OTYM}y!1/&`KF\",<^KY[2u\"|8F^[<zi?NAwqWeycj\"$oHBqy*)C^<PIl4$1W+>Bw\"Q>&?QvA}ghLYOEETL^9"
    "]Ny)RO5RjnmE<fLQRktB=@p.8hq}%&4L9yC\"xQIJz_|Xl=ODWB1K3L[k>D0BX^AtHtPX8~m[K#C^C?Iv:,2t&GAt#BE55Gkq0|t5PSOzy1aLu&$At~w`hq&)"
    "QJ,/bCER^|TK/vV&d6X)yhO%Xse_\"{]Cq?O?IMoQuKFBy|Z\"\"~PA+>dG2|e(g=cL4_\"M>u&APBuBxR0OAA5FKFzyTL0}(kGY$A^E6v8B\"}\"C!(DATwK4{Lv/"
    "]h*rr()hme)B=VOQU_Ms8A!p]aW+$~pCsqV@^WnnT&iWXL9pg(m(gAYVsg4Lws{Tn(dA/YkUxR;y5yaL0Kya1uAA6p(v]2@_CI}Q6>UXiw0:i=KBzKU#ZxhA"
    "i\"O?33yyNl9~p].TqW{EF|wdh=JQKtk+D}?Eoy>T.v3EZ1YgzuDao_%CfstLO?!W}%n_Dt5CIN7L>v#},<8^HlW+%(wJl|~}|X&_UnQ+Kvi`<CMZpiJ`k_^X"
    "sI7FW4p(KKrV5IvrqDH?W|o}f_YFG\"/5cF2_^X;)|K)^~`TsAMF4hdt1U)DDdWJVWMKCt)t(!==pD|0[m?PvkBvW<@1]4$PLH|2q;sWuTKJlzsn?\"Fq?2W<)"
    "Q;<{wyh~p>5]L#I<(^=C]v6P&>GLgC~)sizneW?T@Qgq%}NtHQ/C{ULXwQE|{(UC<Q|ITx;A<yrqliT(Tq98q?UJ?~A{eLz|4CVZWLJ?+1Mpl18K!FkBxi4E"
    "jA2iN`[_k|WXige4KVDvCB+y>}[>m`mFQ*LAYuXo8AC\"BW{EZy\"CI@{9Yl(XP@~4{Ic#J>a*%6HY%CF`Qw~o^L.>{[^UG*g>isX+>&AN*3bEO2QQ_vBV=)k\""
    "p}D<4SLJTqb~XG$w9XZqj~Ft&vo@yJW|ySU)2(HnDHyPi_(qHBg~G/|uu^1i_cuqYV15pA4$b4>X]h/*dZtVS|6X`N+:{s}SrfNd@MT*M*OFLF%`(MzFS\"e?"
    "3Q2[`{hBYKr1;X7<[E<FitS&!~$qKS73\"DQw,TG>u/St\"X0Y=|41.rb@_RJo<)1KuWSt^XHv$^AABW&.$gHwXL>c`HJ&$<QJ!~R.S&<?OD8Sqy4)aFdcd@`Q"
    "3sADlM}Q+Cn(bLzF=FF)(MT?Qnr885QL>F?aWjdM$~p&0A&\"9*aRz|r#G>U;{>^~~=@?|yh:`2uQ06n!T@H&f]F+jiZF61B.Q}5GUq&v.J6yHzLw]WEAT\"[c"
    "fzvvff0@oO!+>YuU{Uz=2B|L]LrK.^)ABD4YrPKIQm<MtMfJh@*5BMLGnwow[poO6ZcXKW4e<M+5@M&ForKay?BzPv_YEghA/MNM6IO/nz@cVzdwMY&mjAL*"
    "RMWJ]9p=izDzVyhl[phAE*xMpKJYKaBMHG;vMcEgCz0+PaqOa4PmMYOM6Is#)M@cJzdwua[pBzMwnX[T4euZ|)JM5G76HTy?Jzsy{a@c+n&Z6YtVx(tZX@=L"
    "oHVsZ9_c^FUv*dEg#Z\"CzX?P.>tZZ@iMWI7yHTNoKzdv,jxJpOsZsX4WE\"KaiGXLAABtSH?^v(T/G)sEGi?DM\"LzEEvI~*<cS<~BkPLj9cdSAugH?RN5z:k4"
    "}w4K\"6U)&WAv;O[j^|8y~hAI6S7n]JX?<DBgq!,Bp|*B5Dce6C=arXsCs$v(E,BRTBLgw)3RY\"+(ACQ!h93KAXBBRHH?9~HtqA$HwR=xBA5FAA$<]F)AKjK?"
    "OGc#cN`(gqy}=q=hxta&Qtp`4v98j@_F*U9(>]F_upO+%KR?hqjB<)cRZ1t+R>i\"bC%}?_in{GHYYE=QRt]X(J.L1uKfBBsv=TsUHVLGl+uKSK4vB&eiH?G}"
    "A*|;ub;y9M[J2ELLiV/27]uq.ocsW`vqHts1bMLvraFtf}~1h&55Q?K}Lt&>X]J7r|2kBHLFx}`2l&<C1B@gTW}vXgQM..*HRVbvF/99]UPLA\"OGP+R&DA))"
    "m?O`ytqBt~(\"=%E*int~!ABB;vQYv^3};FbyxWOW=~QYl=c^ez%C%VzFj_}w)AznX+7(nS|bX!48e;pp%Sn?+N41!QdBwKUqAYpiFAG+8>_J%0U+XZmWj7eb"
    "mW{]CDMf<g~~FuBVn4KVw|mcbs8GHLQt|NUAwujgYA9|`CZL=pP(MVxQ.[+r&5}|@{&n3}7\"7s?v`^8FwFp[E^xqaq0tw@Z].:)Az>q,R~./ZF@bZN&avq_P"
    "&3H?C`c&SvY`@nuu|gC_b2UW[B1(|I3Y*Jz9~CZw2r)&CnBAaM9AD8{9Y4t+PJYMWqaV4}tWOweANH,y]Ng4(>NwPu[2m\">rQJ%BM<sZ=t_Ki|*u.AK4h#Ba"
    ":>{k=Pb;P@bi|oJVrPZyY*|LYGy_kZ:j}}%q#rKC*`jq.adB6E0_Jt<JRR7CeuhBAGC|\"X%h7FRtluDAV|=~%66FSAyWR|q1v$b;;.#{nw9hQWL?\"*EtCFD\""
    "j?7_|>iaq~(.~1KYoUE_Vk^<piQ{ayQ*L}aFbiQVP3$\"UZ}~?}$n(X;>&~yn#W7fgGrV?RxMAGit4}`>v&8C%$$Aiq)vzDG\"3}#@YF5vLUDCdNuDS&u?(..h"
    "MB%5BG<y))x%1~EI[}\"LZFewz84pocFwpnXLBLxq6Cm{S/f{^;]&E,\"{5y0[2L|n?z\"|!^Q(Y}aLiGi|w}}+T:N`BXN&tQM?4XZCyE(FO+i?.^u~MRWK/Rx|"
    "HqWeX(5qQ+eLF^?D{~0Y:>|p_CK&Ti{vrFHL7=^^E~\"+UFb1Kd!VjBn|i|*W>Ko[GWVJ^L9vU|:fsBvs!rIh4KRz+THA<v[9B\"]9m4;yL@D\"2(]LW:U4`%nj"
    "]_IyDS4M]K]k$T%*D\"{TKCbFf{rv:>@|4ipCf6YL{nYVg~;>k|1(uWKEhq?GG>cF8y3+ZB7/fCurBtUXpF?}[eeA.o::BGQwEc[XO>3C8cr3(BpF?)uu#}>s"
    "wFx%3@)ksW;rzKlqnFr?LX0IQ1hK6}2FJVQViIEq7SG>.`I`#GQ4o?dAl[z:*q|DMv3YezSCL>6F&[Jtn?N`51b:53XKyRt!8<H.bJeB$@FGUwpSs~QVtvX+"
    "B\"!FNwk)dB2W5y>X?LT?[D;)f?o/}eTqo=_F1h}lBh+>9vnut]CFX~5CE@BGWqV%PXp/F1=W=~MPQq9_\"4YLWLnu7s/UiL:`~l4?Vn%QZBtPmI@ClsB\"_0r\""
    "i^34NEJ*K3tCqwT|o?A~RVL4LQ_B>r[2B\"BrEU!>)^r?eYQV;`h9_XZ~L{4vQttZp?gqa|yD#FUtlu/!dAB*vUdbmIg(8A<~I*QVuWZFF|?@2F5pDCFhR?He"
    "Ou}Uv?f|ItgNA\"XA}4|Q@9$$qD,buyo&=t*\"$(xt+>s~L|;AG_O^f~%_LC<UFQrRK<`U*5PW3FNxX4h_#^0s;H>K5v=M!WQ.X4buHjCG44n+#Mrb3_{~3k[E"
    "^^^X?juVpFmBI)eRkt`$#[0L=w+~B*ShJ]c#qK7fLJYVRquB20x}/*U/KOrV#]J_H6k+zYYKyq~:T~Y`Ut`OFM|(>s+$Mu5lCGUSE,K_BLC)vUcGHLdZj4z{"
    "A\".wqK2GaFF{zU^Lpy]vlinW(q{~X)g~p1.Dp[n\"#FDLO:z|lZ{>FGiAxM!GknY@KX9~LI&Cs(y/l4f^=VUWbyc#{XmGrc?2x11FF4h&IhY)PW)XrPv`O<7X"
    "]XN?Z47yR@@bZvGxu37LnIc)(jTR*68TCXX*+>/`q?c`.;a|q#&#bwp##,a^LLhq^jkG<C`rJYFV1L+zh%~GjG3}4YtXuw|9\"C4}1kHwbu>Ee]vZmWQI31E%"
    "HMSQGygY$A$tFxz}[Wk|8x4F=b?~e+e6M?AG~Q(>&~?Nx&VCc!m?~`(HY4RqJt!f/\"~CVNF`0tnd=&%~bF!Bzk4qkq9)IY,?VqI!Ah(^Y]:sAMQGJi2T/5gB"
    "RtnusgQe]|WBSCK_g\"~t|8k6v+7k&hIfrPEAA(j#6DEm2nor>]0F=shd(tD6}15yK]CH{lPR*htV|[O*l>zQR|wqTYt?8^EZQ+!hft*5iu9FQvqF@)s`t2Hg"
    "5WeG/s3$6qt?nli#[~;_cD0T_4FCQG4VCWCMxN2Zq(]QxD*TeWIcKvlWc@WMcseul~+J|e1(PAMn;]UJzLvw@9jg[D61h|w*E`*[]vPL\"Kw_!QeXhbLoq$tK"
    "dNRw#I)A%q/XAAmLkqq?cF%tOACG1[8~`~\"^:{C&Ik$M1zg&<Vq@6I~X~ZR9TkK/=>Z:pyJtz5I?2.&)XgI_~v*TxBB^Nsscnrm~Tt0u&Z0(m<Qa.e=1KIX\""
    "boyH9!fbjGn<$wVd[9,xTmfY.Ofdzi.m0zw0Z7>KFgrA3j5Oxd7qr=DdlHD8KWlAgAAAAAAAAAAAAArVAAAAAAAAAAAAAAAAAAAAAAAAAAAAWLAAAAT|OABt"
    "gFAAAAAA|;fABt^GA0AAAA5F/tyFAAAAAAAAAAAAAAINAAAAAAAAAAAAAAeqAAAAAAAAAAAAAAAAAA5FPAAAAAAAAAAAAAAA$A$AAAggxXMIV!b,~))y|D@g"
    "l[r1i/=BbQ1n,u:TtX~H`&]y0(0_^D#dWGEHZ\"KiJIg!!2+tRtME)K&{<8cOAvmFoqZV>,*?\"H]S}wtIlRxB&frjo>d7yYeP*l(:$5n)`H+$?j!Fu\"pYwQ`r"
    "}r$5}AKEZgR=3[<FuYmmBtAAAABtAACAAA~F\"AuWAAAAAAQA$A:CuWAAAAAAgA$A;vXKAABAD^MA^X0EC\"bBEA{>2A5F6InzEAQABt\"A~CAACAIA44SAXLuB"
    "/`JAgAAAAAuWAAAA:&LAAA5FAAAAAAAAAAAA;vifAABAAAAAAA:CuWAA7FAAAAYAzWTAAAv(#c(IAkqQin)HY/c``z*yB/5!fi(,QeRrGk/rU@6LJB(:,$,c"
    "\"Efk*%xD4AC+}TpA4jz2r!bxlgb,JL6B+u`M`CL]GAAAAAAAAAAAoMcxH2vbMo.ciTamsC<T3)jKT>%X*2~=r^Yqv<Wb<l=hIB8=#YqS^4VM^ODSw1h+0QEG"
    "`n^aXgP?~72g,jZQ2b$z(r*i[0E8kE_jh<}CNUxfM`ySOC5QzkE3U&L5Czk;4MTj\"y)CAV(*7][hU5aR>1bq]OHBt17%g0)^#L],zktP2|9Bm6:Q%ldk\"@>G"
    ">2+>C\"^AyU;i1I8((D>Ij!FxOfi:WEM!}u~O)g.jms/RIHy5%[kA((NYlNhM{R~@B5RG{t(L})Zf+CJTx%P2*FjMO2AAXLHXPAiJfrc]?F.v:a|||[m7LatA"
    "#WGme]7/Ry+yKua!\"[\"Z]dM]|hH{;6KEGtmG{|M.;yHBZB7L57Hbmn{<@5a*yO9e{J5?>G`2j?}Eg*[(:kwY%QnZ}O>o+LuEel4@ZX/Fpu7L=FG%dOJXXFAm"
    ">m0dX5tCFYc)b]a/HxlN@sfmQ7bXWS_CxguB\"46FAD^C_|[yvvZZj(_:LwzZSgeEBd{)v4VD?m,z$<NGDzHBX3Sp=+:a/jlfnE(uP7TJ%)hF,t7cSx@inn4x"
    "{[6YgQ~jb&vBRM)H~)i],ACwbXiG64aCIH45aLb\",H!SjD^aClTN*:;dxNtT~8WpbJi::33^X|^8\"ZkcUuPZ[5+BsOWZ0j=,V,{:Z7Cia[M6>D(e";

vector<unsigned char> decode_base91(const string& input) {
    array<int, 256> decode;
    decode.fill(-1);
    for (int i = 0; i < 91; ++i) {
        decode[static_cast<unsigned char>(kBase91Alphabet[i])] = i;
    }

    vector<unsigned char> out;
    int value = -1;
    unsigned int bit_queue = 0;
    int bit_count = 0;
    for (unsigned char ch : input) {
        int decoded = decode[ch];
        if (decoded < 0) continue;
        if (value < 0) {
            value = decoded;
        } else {
            value += decoded * 91;
            bit_queue |= static_cast<unsigned int>(value) << bit_count;
            bit_count += (value & 8191) > 88 ? 13 : 14;
            do {
                out.push_back(static_cast<unsigned char>(bit_queue & 255));
                bit_queue >>= 8;
                bit_count -= 8;
            } while (bit_count > 7);
            value = -1;
        }
    }
    if (value >= 0) {
        out.push_back(static_cast<unsigned char>((bit_queue | (value << bit_count)) & 255));
    }
    return out;
}


using torch::Tensor;

constexpr int MODEL_CHANNELS = 64;
constexpr int MODEL_BLOCKS = 2;
constexpr int MODEL_GROUPS = 8;
constexpr double RESIDUAL_BRANCH_SCALE = 1.0 / MODEL_BLOCKS;

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t groups,
    int64_t padding
) {
    namespace F = torch::nn::functional;
    return F::conv2d(
        input,
        weight,
        F::Conv2dFuncOptions().bias(bias).groups(groups).padding(padding)
    );
}

Tensor qlinear(const Tensor& x, const Tensor& weight, const Tensor& bias = Tensor()) {
    Tensor y = torch::matmul(x, weight.t());
    return bias.defined() ? y + bias : y;
}

Tensor group_norm(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    const auto sizes = input.sizes();
    Tensor grouped = input.view({sizes[0], MODEL_GROUPS, MODEL_CHANNELS / MODEL_GROUPS, sizes[2], sizes[3]});
    Tensor mean = grouped.mean({2, 3, 4}, true);
    Tensor centered = grouped - mean;
    Tensor variance = (centered * centered).mean({2, 3, 4}, true);
    Tensor normalized = centered / torch::sqrt(variance + 1e-5);
    return normalized.view_as(input) * weight.view({1, MODEL_CHANNELS, 1, 1})
        + bias.view({1, MODEL_CHANNELS, 1, 1});
}

struct ConvNeXtBlock {
    Tensor layer_scale;
    Tensor depthwise_weight, depthwise_bias;
    Tensor norm_weight, norm_bias;
    Tensor pointwise1_weight, pointwise1_bias;
    Tensor pointwise2_weight, pointwise2_bias;

    Tensor forward(const Tensor& input) const {
        Tensor residual = input;
        Tensor y = conv2d(input, depthwise_weight, depthwise_bias, MODEL_CHANNELS, 1);
        y = y.permute({0, 2, 3, 1});
        Tensor mean = y.mean(-1, true);
        Tensor centered = y - mean;
        Tensor variance = (centered * centered).mean(-1, true);
        y = centered / torch::sqrt(variance + 1e-5);
        y = y * norm_weight + norm_bias;
        y = qlinear(y, pointwise1_weight, pointwise1_bias);
        y = 0.5 * y * (1.0 + torch::erf(y / std::sqrt(2.0)));
        y = qlinear(y, pointwise2_weight, pointwise2_bias);
        y = y * layer_scale;
        y = y.permute({0, 3, 1, 2});
        return (1.0 - RESIDUAL_BRANCH_SCALE) * residual
            + RESIDUAL_BRANCH_SCALE * y;
    }
};

struct Q4Reader {
    const vector<unsigned char>& data;
    size_t pos = 0;
    explicit Q4Reader(const vector<unsigned char>& source) : data(source) {
        const char magic[] = "AHC061Q4";
        for (int i = 0; i < 8; ++i) {
            if (data.at(pos++) != magic[i]) throw runtime_error("bad q4 model");
        }
        if (data.at(pos++) != 1) throw runtime_error("unsupported q4 model");
        pos += 2;  // tensor count; the fixed actor layout below validates consumption.
    }
    uint32_t u32() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value |= uint32_t(data.at(pos++)) << (8 * i);
        return value;
    }
    float half() {
        uint16_t bits = uint16_t(data.at(pos)) | (uint16_t(data.at(pos + 1)) << 8);
        pos += 2;
        c10::Half value;
        memcpy(&value, &bits, sizeof(bits));
        return static_cast<float>(value);
    }
    Tensor take(vector<int64_t> shape) {
        const int mode = data.at(pos++);
        const uint32_t n = u32();
        int64_t expected = 1;
        for (int64_t d : shape) expected *= d;
        if (n != expected) {
            throw runtime_error(
                "q4 tensor shape mismatch: got " + to_string(n) + " expected " + to_string(expected)
            );
        }
        vector<float> values(n);
        if (mode == 1) {
            for (uint32_t i = 0; i < n; ++i) values[i] = half();
        } else if (mode == 0) {
            const size_t codes = (n + 1) / 2;
            const size_t scales = (n + 127) / 128;
            const size_t code_start = pos;
            pos += codes;
            vector<float> group_scales(scales);
            for (size_t i = 0; i < scales; ++i) group_scales[i] = half();
            for (uint32_t i = 0; i < n; ++i) {
                int q = (data[code_start + i / 2] >> (4 * (i & 1))) & 15;
                if (q >= 8) q -= 16;
                values[i] = q * group_scales[i / 128];
            }
        } else throw runtime_error("bad q4 tensor mode");
        return torch::from_blob(values.data(), shape, torch::kFloat32).clone();
    }
};

struct Q4Policy {
    Tensor trunk_weight, trunk_norm_weight, trunk_norm_bias;
    vector<ConvNeXtBlock> blocks;
    Tensor policy_weight, policy_norm_weight, policy_norm_bias;
    Tensor readout_weight, readout_gain;
    Tensor mean, invstd;
    explicit Q4Policy(const vector<unsigned char>& bytes) {
        Q4Reader reader(bytes);
        trunk_weight = reader.take({MODEL_CHANNELS, NUM_PLANES, 3, 3});
        trunk_norm_weight = reader.take({MODEL_CHANNELS});
        trunk_norm_bias = reader.take({MODEL_CHANNELS});
        for (int i = 0; i < MODEL_BLOCKS; ++i) {
            ConvNeXtBlock block;
            block.layer_scale = reader.take({MODEL_CHANNELS});
            block.depthwise_weight = reader.take({MODEL_CHANNELS, 1, 3, 3});
            block.depthwise_bias = reader.take({MODEL_CHANNELS});
            block.norm_weight = reader.take({MODEL_CHANNELS});
            block.norm_bias = reader.take({MODEL_CHANNELS});
            block.pointwise1_weight = reader.take({MODEL_CHANNELS * 4, MODEL_CHANNELS});
            block.pointwise1_bias = reader.take({MODEL_CHANNELS * 4});
            block.pointwise2_weight = reader.take({MODEL_CHANNELS, MODEL_CHANNELS * 4});
            block.pointwise2_bias = reader.take({MODEL_CHANNELS});
            blocks.push_back(move(block));
        }
        policy_weight = reader.take({MODEL_CHANNELS, MODEL_CHANNELS, 1, 1});
        policy_norm_weight = reader.take({MODEL_CHANNELS});
        policy_norm_bias = reader.take({MODEL_CHANNELS});
        readout_weight = reader.take({1, MODEL_CHANNELS, 1, 1});
        readout_gain = reader.take({1});
        mean = reader.take({1, NUM_PLANES, 1, 1});
        invstd = reader.take({1, NUM_PLANES, 1, 1});
        if (reader.pos != bytes.size()) throw runtime_error("q4 model has trailing data");
    }
    Tensor forward(const Tensor& input) const {
        Tensor raw = input.to(torch::kFloat32);
        Tensor x = (raw - mean) * invstd;
        x = group_norm(conv2d(x, trunk_weight, Tensor(), 1, 1), trunk_norm_weight, trunk_norm_bias);
        x = torch::relu(x);
        for (const auto& block : blocks) x = block.forward(x);
        x = group_norm(conv2d(x, policy_weight, Tensor(), 1, 0), policy_norm_weight, policy_norm_bias);
        x = torch::relu(x);
        x = conv2d(x, readout_weight, Tensor(), 1, 0) * readout_gain;
        return x.reshape({1, N * N});
    }
};

struct State {
    int m = 0;
    int u = 0;
    int turn = 0;
    int values[N][N]{};
    int owner[N][N]{};
    int level[N][N]{};
    pair<int, int> pos[MAX_PLAYERS]{};
};

vector<pair<int, int>> get_candidates(const State& st, int player) {
    vector<pair<int, int>> reachable;
    bool visited[N][N]{};
    queue<pair<int, int>> q;
    q.push(st.pos[player]);
    visited[st.pos[player].first][st.pos[player].second] = true;

    constexpr int dx[4] = {0, 1, 0, -1};
    constexpr int dy[4] = {1, 0, -1, 0};
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        bool ok = true;
        for (int p = 0; p < st.m; ++p) {
            if (p != player && st.pos[p] == make_pair(x, y)) {
                ok = false;
                break;
            }
        }
        if (ok) reachable.push_back({x, y});
        if (st.owner[x][y] != player) continue;
        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d];
            int ny = y + dy[d];
            if (0 <= nx && nx < N && 0 <= ny && ny < N && !visited[nx][ny]) {
                visited[nx][ny] = true;
                q.push({nx, ny});
            }
        }
    }
    return reachable;
}

array<unsigned char, N * N> connected_component_mask(const State& st, int player) {
    array<unsigned char, N * N> mask{};
    auto [sx, sy] = st.pos[player];
    if (st.owner[sx][sy] != player) return mask;

    queue<pair<int, int>> q;
    mask[sx * N + sy] = 1;
    q.push({sx, sy});
    constexpr int dx[4] = {0, 1, 0, -1};
    constexpr int dy[4] = {1, 0, -1, 0};
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d];
            int ny = y + dy[d];
            bool in_bounds = 0 <= nx && nx < N && 0 <= ny && ny < N;
            if (in_bounds && !mask[nx * N + ny] && st.owner[nx][ny] == player) {
                mask[nx * N + ny] = 1;
                q.push({nx, ny});
            }
        }
    }
    return mask;
}

array<float, MAX_PLAYERS> player_scores(const State& st) {
    array<float, MAX_PLAYERS> scores{};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int p = st.owner[i][j];
            if (p >= 0) scores[p] += static_cast<float>(st.values[i][j] * st.level[i][j]);
        }
    }
    return scores;
}

array<int, MAX_PLAYERS> player_id_map(const array<float, MAX_PLAYERS>& scores, int m) {
    vector<int> enemies;
    for (int p = 1; p < m; ++p) enemies.push_back(p);
    sort(enemies.begin(), enemies.end(), [&](int a, int b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    array<int, MAX_PLAYERS> mapped;
    mapped.fill(0);
    mapped[0] = 0;
    for (int i = 0; i < static_cast<int>(enemies.size()); ++i) mapped[enemies[i]] = i + 1;
    return mapped;
}

struct Particle {
    double wa = 0.0;
    double wb = 0.0;
    double wc = 0.0;
    double wd = 0.0;
    double eps = 0.0;
};

struct SplitMix64 {
    uint64_t state;
    bool has_spare = false;
    double spare = 0.0;

    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next_u64() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    double next_f64() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / static_cast<double>(1ULL << 53));
    }

    double uniform(double low, double high) {
        return low + (high - low) * next_f64();
    }

    double normal() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        double u1 = max(next_f64(), numeric_limits<double>::min());
        double u2 = next_f64();
        double radius = sqrt(-2.0 * log(u1));
        double theta = 2.0 * acos(-1.0) * u2;
        spare = radius * sin(theta);
        has_spare = true;
        return radius * cos(theta);
    }
};

double particle_score(const State& st, int player, int x, int y, const Particle& particle) {
    int owner = st.owner[x][y];
    int level = st.level[x][y];
    double value = static_cast<double>(st.values[x][y]);
    if (owner == -1) return value * particle.wa;
    if (owner == player) return level < st.u ? value * particle.wb : 0.0;
    if (level == 1) return value * particle.wc;
    return value * particle.wd;
}

void add_policy_distribution(
    const State& st,
    int player,
    const vector<pair<int, int>>& candidates,
    const Particle& particle,
    double weight,
    array<double, N * N>& dist
) {
    if (candidates.empty()) return;
    double random_prob = particle.eps / static_cast<double>(candidates.size());
    for (auto [x, y] : candidates) dist[x * N + y] += weight * random_prob;

    vector<double> scores;
    scores.reserve(candidates.size());
    double best_score = -numeric_limits<double>::infinity();
    for (auto [x, y] : candidates) {
        double score = particle_score(st, player, x, y, particle);
        best_score = max(best_score, score);
        scores.push_back(score);
    }
    double tolerance = 1e-9 * max(abs(best_score), 1.0);
    int best_count = 0;
    for (double score : scores) {
        if (score >= best_score - tolerance) ++best_count;
    }
    best_count = max(best_count, 1);
    double greedy_prob = (1.0 - particle.eps) / static_cast<double>(best_count);
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        if (scores[i] >= best_score - tolerance) {
            auto [x, y] = candidates[i];
            dist[x * N + y] += weight * greedy_prob;
        }
    }
}

struct ParticleFilterSmc {
    vector<Particle> particles;
    vector<double> weights;
    SplitMix64 rng;

    ParticleFilterSmc(int n, uint64_t seed) : rng(seed) {
        n = max(n, 1);
        particles.reserve(n);
        for (int i = 0; i < n; ++i) {
            particles.push_back(Particle{
                rng.uniform(0.3, 1.0),
                rng.uniform(0.3, 1.0),
                rng.uniform(0.3, 1.0),
                rng.uniform(0.3, 1.0),
                rng.uniform(0.1, 0.5),
            });
        }
        weights.assign(n, 1.0 / static_cast<double>(n));
    }

    Particle mean() const {
        Particle m;
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
            m.wa += weights[i] * particles[i].wa;
            m.wb += weights[i] * particles[i].wb;
            m.wc += weights[i] * particles[i].wc;
            m.wd += weights[i] * particles[i].wd;
            m.eps += weights[i] * particles[i].eps;
        }
        return m;
    }

    Particle stddev(const Particle& m) const {
        Particle v;
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
            v.wa += weights[i] * pow(particles[i].wa - m.wa, 2);
            v.wb += weights[i] * pow(particles[i].wb - m.wb, 2);
            v.wc += weights[i] * pow(particles[i].wc - m.wc, 2);
            v.wd += weights[i] * pow(particles[i].wd - m.wd, 2);
            v.eps += weights[i] * pow(particles[i].eps - m.eps, 2);
        }
        return Particle{sqrt(max(v.wa, 0.0)), sqrt(max(v.wb, 0.0)), sqrt(max(v.wc, 0.0)),
                        sqrt(max(v.wd, 0.0)), sqrt(max(v.eps, 0.0))};
    }

    double ess() const {
        double sum_sq = 0.0;
        for (double w : weights) sum_sq += w * w;
        return sum_sq <= 0.0 ? 0.0 : 1.0 / sum_sq;
    }

    void update(const State& st, int player, pair<int, int> observed) {
        vector<pair<int, int>> candidates = get_candidates(st, player);
        auto it = find(candidates.begin(), candidates.end(), observed);
        if (it == candidates.end() || candidates.empty()) return;
        int obs_idx = static_cast<int>(it - candidates.begin());

        vector<double> logs;
        logs.reserve(particles.size());
        double max_log = -numeric_limits<double>::infinity();
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
            array<double, N * N> dist{};
            add_policy_distribution(st, player, candidates, particles[i], 1.0, dist);
            auto [x, y] = candidates[obs_idx];
            double prob = max(dist[x * N + y], 1e-300);
            double log_w = log(max(weights[i], 1e-300)) + log(prob);
            max_log = max(max_log, log_w);
            logs.push_back(log_w);
        }

        double sum = 0.0;
        for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
            weights[i] = exp(logs[i] - max_log);
            sum += weights[i];
        }
        if (!isfinite(sum) || sum <= 0.0) {
            fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(weights.size()));
            return;
        }
        for (double& w : weights) w /= sum;
        if (ess() < 0.5 * static_cast<double>(particles.size())) resample();
    }

    void resample() {
        int n = static_cast<int>(particles.size());
        Particle m = mean();
        Particle s = stddev(m);
        vector<double> cumulative(n);
        partial_sum(weights.begin(), weights.end(), cumulative.begin());
        cumulative.back() = 1.0;
        double step = 1.0 / static_cast<double>(n);
        double u = rng.next_f64() * step;
        double a = 0.98;
        double h = sqrt(1.0 - a * a);
        int idx = 0;
        vector<Particle> next;
        next.reserve(n);
        auto jitter = [&](double value, double mean_value, double sd, double low, double high) {
            double center = a * value + (1.0 - a) * mean_value;
            return min(high, max(low, center + h * sd * rng.normal()));
        };
        for (int i = 0; i < n; ++i) {
            while (idx + 1 < n && cumulative[idx] < u) ++idx;
            Particle p = particles[idx];
            next.push_back(Particle{
                jitter(p.wa, m.wa, s.wa, 0.3, 1.0),
                jitter(p.wb, m.wb, s.wb, 0.3, 1.0),
                jitter(p.wc, m.wc, s.wc, 0.3, 1.0),
                jitter(p.wd, m.wd, s.wd, 0.3, 1.0),
                jitter(p.eps, m.eps, s.eps, 0.1, 0.5),
            });
            u += step;
        }
        particles = move(next);
        fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(n));
    }

    array<float, N * N> predictive_distribution(const State& st, int player) const {
        array<double, N * N> tmp{};
        vector<pair<int, int>> candidates = get_candidates(st, player);
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
            add_policy_distribution(st, player, candidates, particles[i], weights[i], tmp);
        }
        array<float, N * N> out{};
        for (int i = 0; i < N * N; ++i) out[i] = static_cast<float>(tmp[i]);
        return out;
    }
};

array<unsigned char, N * N> reach_mask(const State& st, int player) {
    array<unsigned char, N * N> mask{};
    for (auto [x, y] : get_candidates(st, player)) mask[x * N + y] = 1;
    return mask;
}

array<float, N * N> dist_to_sources(const array<unsigned char, N * N>& sources) {
    constexpr int INF = 1 << 20;
    array<int, N * N> dist;
    dist.fill(INF);
    bool has_source = false;
    for (int idx = 0; idx < N * N; ++idx) {
        if (sources[idx]) {
            dist[idx] = 0;
            has_source = true;
        }
    }
    array<float, N * N> out{};
    if (!has_source) {
        out.fill(1.0f);
        return out;
    }
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            int idx = x * N + y;
            if (x > 0) dist[idx] = min(dist[idx], dist[(x - 1) * N + y] + 1);
            if (y > 0) dist[idx] = min(dist[idx], dist[x * N + y - 1] + 1);
        }
    }
    for (int x = N - 1; x >= 0; --x) {
        for (int y = N - 1; y >= 0; --y) {
            int idx = x * N + y;
            if (x + 1 < N) dist[idx] = min(dist[idx], dist[(x + 1) * N + y] + 1);
            if (y + 1 < N) dist[idx] = min(dist[idx], dist[x * N + y + 1] + 1);
        }
    }
    for (int idx = 0; idx < N * N; ++idx) out[idx] = dist[idx] >= INF ? 1.0f : dist[idx] / 18.0f;
    return out;
}

torch::Tensor encode(
    const State& st,
    const vector<pair<int, int>>& candidates,
    const vector<ParticleFilterSmc>& pfilters
) {
    vector<float> planes(NUM_PLANES * N * N, 0.0f);
    auto at = [&](int plane, int x, int y) -> float& {
        return planes[(plane * N + x) * N + y];
    };

    float mean_values = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) mean_values += st.values[i][j];
    }
    mean_values /= static_cast<float>(N * N);
    mean_values = max(mean_values, 1.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) at(0, i, j) = st.values[i][j] / mean_values;
    }

    auto scores = player_scores(st);
    auto mapped = player_id_map(scores, st.m);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int owner = st.owner[i][j];
            int mapped_owner = owner < 0 ? -1 : mapped[owner];
            at(mapped_owner + 2, i, j) = 1.0f;
            int lv = st.level[i][j];
            if (1 <= lv && lv <= MAX_LEVEL) at(9 + lv, i, j) = 1.0f;
        }
    }

    for (int p = 0; p < st.m; ++p) {
        int mp = mapped[p];
        auto [x, y] = st.pos[p];
        if (0 <= mp && mp < MAX_PLAYERS && 0 <= x && x < N && 0 <= y && y < N) {
            at(15 + mp, x, y) = 1.0f;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            at(23, i, j) = (100.0f - static_cast<float>(st.turn)) / 100.0f;
            at(PLANE_M, i, j) = static_cast<float>(st.m) / MAX_PLAYERS;
            at(PLANE_U, i, j) = static_cast<float>(st.u) / MAX_LEVEL;
        }
    }

    float player0_score = scores[0];
    float max_ai_score = 0.0f;
    for (int p = 1; p < st.m; ++p) max_ai_score = max(max_ai_score, scores[p]);
    float total_capacity = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) total_capacity += st.values[i][j] * max(st.u, 1);
    }
    total_capacity = max(total_capacity, 1.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            at(PLANE_SCORE_RATIO, i, j) = player0_score / max(max_ai_score, 1.0f);
            at(PLANE_SCORE_DIFF, i, j) = (player0_score - max_ai_score) / total_capacity;
        }
    }

    for (auto [x, y] : candidates) at(PLANE_LEGAL_MASK, x, y) = 1.0f;
    for (int p = 0; p < st.m; ++p) {
        int mp = mapped[p];
        if (0 <= mp && mp < MAX_PLAYERS) {
            float normalized_score = scores[p] / total_capacity;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    at(PLANE_PLAYER_SCORE_START + mp, i, j) = normalized_score;
                }
            }
            int param_start = PLANE_ORACLE_PARAM_START + mp * ORACLE_PARAMS_PER_PLAYER;
            Particle params = p > 0 ? pfilters[p - 1].mean() : Particle{};
            array<float, ORACLE_PARAMS_PER_PLAYER> values{
                static_cast<float>(params.wa),
                static_cast<float>(params.wb),
                static_cast<float>(params.wc),
                static_cast<float>(params.wd),
                static_cast<float>(params.eps),
            };
            for (int k = 0; k < ORACLE_PARAMS_PER_PLAYER; ++k) {
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) at(param_start + k, i, j) = values[k];
                }
            }
        }
    }

    vector<array<unsigned char, N * N>> comp_masks;
    vector<array<unsigned char, N * N>> reach_masks;
    vector<array<float, N * N>> next_planes;
    comp_masks.reserve(st.m);
    reach_masks.reserve(st.m);
    next_planes.reserve(st.m);
    for (int p = 0; p < st.m; ++p) {
        comp_masks.push_back(connected_component_mask(st, p));
        reach_masks.push_back(reach_mask(st, p));
        if (p == 0) {
            array<float, N * N> own_next{};
            for (int idx = 0; idx < N * N; ++idx) {
                own_next[idx] = reach_masks.back()[idx] ? 1.0f : 0.0f;
            }
            next_planes.push_back(own_next);
        } else {
            next_planes.push_back(pfilters[p - 1].predictive_distribution(st, p));
        }
    }

    for (int p = 0; p < st.m; ++p) {
        int mp = mapped[p];
        if (mp < 0 || mp >= MAX_PLAYERS) continue;
        array<unsigned char, N * N> owner_sources{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) owner_sources[i * N + j] = st.owner[i][j] == p;
        }
        auto dist_owner = dist_to_sources(owner_sources);
        auto dist_comp = dist_to_sources(comp_masks[p]);
        for (int idx = 0; idx < N * N; ++idx) {
            int i = idx / N;
            int j = idx % N;
            at(PLANE_COMP_START + mp, i, j) = comp_masks[p][idx] ? 1.0f : 0.0f;
            at(PLANE_REACH_START + mp, i, j) = reach_masks[p][idx] ? 1.0f : 0.0f;
            at(PLANE_NEXT_GREEDY_START + mp, i, j) = next_planes[p][idx];
            at(PLANE_DIST_OWNER_START + mp, i, j) = dist_owner[idx];
            at(PLANE_DIST_COMP_START + mp, i, j) = dist_comp[idx];
        }
    }

    const float inv_board_span = 1.0f / static_cast<float>(N - 1);
    const float pos0_x_norm = static_cast<float>(st.pos[0].first) * inv_board_span;
    const float pos0_y_norm = static_cast<float>(st.pos[0].second) * inv_board_span;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float dx = abs(static_cast<float>(i) - 4.5f);
            float dy = abs(static_cast<float>(j) - 4.5f);
            at(PLANE_DIST_CENTER, i, j) = (dx + dy) / 9.0f;
            at(PLANE_X_NORM, i, j) = static_cast<float>(i) * inv_board_span;
            at(PLANE_Y_NORM, i, j) = static_cast<float>(j) * inv_board_span;
            at(PLANE_POS0_X_NORM, i, j) = pos0_x_norm;
            at(PLANE_POS0_Y_NORM, i, j) = pos0_y_norm;
        }
    }

    const float level_capacity = max(static_cast<float>(N * N * max(st.u, 1)), 1.0f);
    for (int p = 0; p < st.m; ++p) {
        int mp = mapped[p];
        if (mp < 0 || mp >= MAX_PLAYERS) continue;
        float owner_level_sum = 0.0f;
        float owner_level_value_sum = 0.0f;
        float comp_level_sum = 0.0f;
        float comp_level_value_sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float level = static_cast<float>(st.level[i][j]);
                float level_value = level * static_cast<float>(st.values[i][j]);
                if (st.owner[i][j] == p) {
                    owner_level_sum += level;
                    owner_level_value_sum += level_value;
                }
                if (comp_masks[p][i * N + j]) {
                    comp_level_sum += level;
                    comp_level_value_sum += level_value;
                }
            }
        }
        int agg_start = PLANE_PLAYER_AGG_START + mp * PLAYER_AGG_FEATURES;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                at(agg_start + PLAYER_AGG_OWNER_LEVEL_SUM, i, j) = owner_level_sum / level_capacity;
                at(agg_start + PLAYER_AGG_OWNER_LEVEL_VALUE_SUM, i, j) =
                    owner_level_value_sum / total_capacity;
                at(agg_start + PLAYER_AGG_COMP_LEVEL_SUM, i, j) = comp_level_sum / level_capacity;
                at(agg_start + PLAYER_AGG_COMP_LEVEL_VALUE_SUM, i, j) =
                    comp_level_value_sum / total_capacity;
            }
        }
    }

    return torch::from_blob(planes.data(), {1, NUM_PLANES, N, N}, torch::kFloat32)
        .to(torch::kBFloat16);
}

Q4Policy load_model() { return Q4Policy(decode_base91(kEncodedModel)); }

pair<int, int> choose_action(
    Q4Policy& module,
    const State& st,
    const vector<ParticleFilterSmc>& pfilters,
    mt19937& rng
) {
    vector<pair<int, int>> candidates = get_candidates(st, 0);
    if (candidates.empty()) return st.pos[0];
    torch::NoGradGuard no_grad;
    torch::Tensor input = encode(st, candidates, pfilters);
    torch::Tensor logits = module.forward(input).reshape({N * N}).contiguous();
    auto acc = logits.accessor<float, 1>();

    if (ACTION_TEMPERATURE <= 0.0) {
        pair<int, int> best = candidates[0];
        float best_logit = -numeric_limits<float>::infinity();
        for (auto [x, y] : candidates) {
            int idx = x * N + y;
            if (acc[idx] > best_logit) {
                best_logit = acc[idx];
                best = {x, y};
            }
        }
        return best;
    }

    float max_logit = -numeric_limits<float>::infinity();
    for (auto [x, y] : candidates) {
        int idx = x * N + y;
        max_logit = max(max_logit, acc[idx]);
    }
    vector<double> weights;
    weights.reserve(candidates.size());
    for (auto [x, y] : candidates) {
        int idx = x * N + y;
        weights.push_back(exp(static_cast<double>(acc[idx] - max_logit) / ACTION_TEMPERATURE));
    }
    discrete_distribution<int> dist(weights.begin(), weights.end());
    return candidates[dist(rng)];
}

}  // namespace

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    at::set_num_threads(1);
    at::set_num_interop_threads(1);

    State st;
    int input_n = 0;
    cin >> input_n >> st.m >> st.turn >> st.u;
    st.turn = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) cin >> st.values[i][j];
    }
    for (int p = 0; p < st.m; ++p) {
        cin >> st.pos[p].first >> st.pos[p].second;
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            st.owner[i][j] = -1;
            st.level[i][j] = 0;
        }
    }
    for (int p = 0; p < st.m; ++p) {
        auto [x, y] = st.pos[p];
        st.owner[x][y] = p;
        st.level[x][y] = 1;
    }

    Q4Policy module = load_model();
    vector<ParticleFilterSmc> pfilters;
    pfilters.reserve(max(st.m - 1, 0));
    for (int p = 1; p < st.m; ++p) {
        pfilters.emplace_back(
            PF_PARTICLES,
            0xa0761d6478bd642fULL ^ (static_cast<uint64_t>(p) << 32)
        );
    }
    mt19937 rng(static_cast<uint32_t>(
        chrono::steady_clock::now().time_since_epoch().count()
    ));

    for (int t = 0; t < T; ++t) {
        st.turn = t;
        auto [x, y] = choose_action(module, st, pfilters, rng);
        cout << x << ' ' << y << endl;

        vector<pair<int, int>> selected(st.m);
        for (int p = 0; p < st.m; ++p) {
            int tx, ty;
            cin >> tx >> ty;
            selected[p] = {tx, ty};
        }
        for (int p = 1; p < st.m; ++p) {
            pfilters[p - 1].update(st, p, selected[p]);
        }
        for (int p = 0; p < st.m; ++p) {
            cin >> st.pos[p].first >> st.pos[p].second;
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) cin >> st.owner[i][j];
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) cin >> st.level[i][j];
        }
    }
    return 0;
}
