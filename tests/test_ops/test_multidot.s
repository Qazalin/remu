
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_10_45_45_65>:
	s_mov_b32 s8, s13                                          // 000000001700: BE88000D
	s_clause 0x1                                               // 000000001704: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001708: F4080100 F8000000
	s_load_b64 s[12:13], s[0:1], 0x10                          // 000000001710: F4040300 F8000010
	s_mul_i32 s2, s14, 0x41                                    // 000000001718: 9602FF0E 00000041
	s_mul_i32 s0, s15, 0xb6d                                   // 000000001720: 9600FF0F 00000B6D
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_ashr_i32 s1, s0, 31                                      // 00000000172C: 86019F00
	s_lshl_b64 s[10:11], s[2:3], 2                             // 000000001730: 848A8202
	s_ashr_i32 s9, s8, 31                                      // 000000001734: 86099F08
	s_lshl_b64 s[2:3], s[0:1], 2                               // 000000001738: 84828200
	v_mov_b32_e32 v0, 0                                        // 00000000173C: 7E000280
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s6, s6, s10                                      // 000000001744: 80060A06
	s_addc_u32 s7, s7, s11                                     // 000000001748: 82070B07
	s_lshl_b64 s[0:1], s[8:9], 2                               // 00000000174C: 84808208
	s_movk_i32 s10, 0x41                                       // 000000001750: B00A0041
	s_add_u32 s8, s12, s0                                      // 000000001754: 8008000C
	s_addc_u32 s9, s13, s1                                     // 000000001758: 8209010D
	s_add_u32 s12, s6, s2                                      // 00000000175C: 800C0206
	s_addc_u32 s13, s7, s3                                     // 000000001760: 820D0307
	s_add_u32 s24, s8, s2                                      // 000000001764: 80180208
	s_addc_u32 s25, s9, s3                                     // 000000001768: 82190309
	s_load_b256 s[16:23], s[12:13], null                       // 00000000176C: F40C0406 F8000000
	s_clause 0x7                                               // 000000001774: BF850007
	s_load_b32 s11, s[24:25], null                             // 000000001778: F40002CC F8000000
	s_load_b32 s26, s[24:25], 0xb4                             // 000000001780: F400068C F80000B4
	s_load_b32 s27, s[24:25], 0x168                            // 000000001788: F40006CC F8000168
	s_load_b32 s28, s[24:25], 0x21c                            // 000000001790: F400070C F800021C
	s_load_b32 s29, s[24:25], 0x2d0                            // 000000001798: F400074C F80002D0
	s_load_b32 s30, s[24:25], 0x384                            // 0000000017A0: F400078C F8000384
	s_load_b32 s31, s[24:25], 0x438                            // 0000000017A8: F40007CC F8000438
	s_load_b32 s33, s[24:25], 0x4ec                            // 0000000017B0: F400084C F80004EC
	s_add_i32 s10, s10, -13                                    // 0000000017B8: 810ACD0A
	s_add_u32 s6, s6, 52                                       // 0000000017BC: 8006B406
	s_addc_u32 s7, s7, 0                                       // 0000000017C0: 82078007
	s_add_u32 s8, s8, 0x924                                    // 0000000017C4: 8008FF08 00000924
	s_addc_u32 s9, s9, 0                                       // 0000000017CC: 82098009
	s_cmp_eq_u32 s10, 0                                        // 0000000017D0: BF06800A
	s_waitcnt lgkmcnt(0)                                       // 0000000017D4: BF89FC07
	v_fmac_f32_e64 v0, s16, s11                                // 0000000017D8: D52B0000 00001610
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E0: BF870091
	v_fmac_f32_e64 v0, s17, s26                                // 0000000017E4: D52B0000 00003411
	v_fmac_f32_e64 v0, s18, s27                                // 0000000017EC: D52B0000 00003612
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000017F4: BF8700B1
	v_fmac_f32_e64 v0, s19, s28                                // 0000000017F8: D52B0000 00003813
	s_load_b128 s[16:19], s[12:13], 0x20                       // 000000001800: F4080406 F8000020
	s_load_b32 s11, s[24:25], 0x5a0                            // 000000001808: F40002CC F80005A0
	v_fmac_f32_e64 v0, s20, s29                                // 000000001810: D52B0000 00003A14
	s_load_b32 s20, s[24:25], 0x654                            // 000000001818: F400050C F8000654
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001820: BF8700A1
	v_fmac_f32_e64 v0, s21, s30                                // 000000001824: D52B0000 00003C15
	s_load_b32 s21, s[24:25], 0x708                            // 00000000182C: F400054C F8000708
	v_fmac_f32_e64 v0, s22, s31                                // 000000001834: D52B0000 00003E16
	s_load_b32 s22, s[24:25], 0x7bc                            // 00000000183C: F400058C F80007BC
	s_load_b32 s12, s[12:13], 0x30                             // 000000001844: F4000306 F8000030
	s_load_b32 s13, s[24:25], 0x870                            // 00000000184C: F400034C F8000870
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001854: BF8700A1
	v_fmac_f32_e64 v0, s23, s33                                // 000000001858: D52B0000 00004217
	s_waitcnt lgkmcnt(0)                                       // 000000001860: BF89FC07
	v_fmac_f32_e64 v0, s16, s11                                // 000000001864: D52B0000 00001610
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000186C: BF870091
	v_fmac_f32_e64 v0, s17, s20                                // 000000001870: D52B0000 00002811
	v_fmac_f32_e64 v0, s18, s21                                // 000000001878: D52B0000 00002A12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001880: BF870091
	v_fmac_f32_e64 v0, s19, s22                                // 000000001884: D52B0000 00002C13
	v_fmac_f32_e64 v0, s12, s13                                // 00000000188C: D52B0000 00001A0C
	s_cbranch_scc0 65457                                       // 000000001894: BFA1FFB1 <r_10_45_45_65+0x5c>
	s_mul_i32 s2, s15, 0x7e9                                   // 000000001898: 9602FF0F 000007E9
	s_mul_i32 s6, s14, 45                                      // 0000000018A0: 9606AD0E
	s_ashr_i32 s3, s2, 31                                      // 0000000018A4: 86039F02
	v_mov_b32_e32 v1, 0                                        // 0000000018A8: 7E020280
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000018AC: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000018B0: BF8704B9
	s_add_u32 s4, s4, s2                                       // 0000000018B4: 80040204
	s_addc_u32 s5, s5, s3                                      // 0000000018B8: 82050305
	s_ashr_i32 s7, s6, 31                                      // 0000000018BC: 86079F06
	s_lshl_b64 s[2:3], s[6:7], 2                               // 0000000018C0: 84828206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018C4: BF870009
	s_add_u32 s2, s4, s2                                       // 0000000018C8: 80020204
	s_addc_u32 s3, s5, s3                                      // 0000000018CC: 82030305
	s_add_u32 s0, s2, s0                                       // 0000000018D0: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000018D4: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000018D8: DC6A0000 00000001
	s_nop 0                                                    // 0000000018E0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018E4: BFB60003
	s_endpgm                                                   // 0000000018E8: BFB00000
