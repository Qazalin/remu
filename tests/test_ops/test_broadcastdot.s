
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_450_45_65>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b64 s[8:9], s[0:1], 0x10                            // 000000001704: F4040200 F8000010
	s_load_b128 s[4:7], s[0:1], null                           // 00000000170C: F4080100 F8000000
	s_mov_b32 s2, s14                                          // 000000001714: BE82000E
	s_ashr_i32 s3, s14, 31                                     // 000000001718: 86039F0E
	s_mul_i32 s10, s15, 0x41                                   // 00000000171C: 960AFF0F 00000041
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001724: 84808202
	s_ashr_i32 s11, s10, 31                                    // 000000001728: 860B9F0A
	v_mov_b32_e32 v0, 0                                        // 00000000172C: 7E000280
	s_waitcnt lgkmcnt(0)                                       // 000000001730: BF89FC07
	s_add_u32 s8, s8, s0                                       // 000000001734: 80080008
	s_addc_u32 s9, s9, s1                                      // 000000001738: 82090109
	s_lshl_b64 s[2:3], s[10:11], 2                             // 00000000173C: 8482820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001740: BF870009
	s_add_u32 s2, s2, s6                                       // 000000001744: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001748: 82030703
	s_add_u32 s2, s2, 48                                       // 00000000174C: 8002B002
	s_addc_u32 s3, s3, 0                                       // 000000001750: 82038003
	s_mov_b64 s[6:7], 0                                        // 000000001754: BE860180
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001758: BF870009
	s_add_u32 s10, s8, s6                                      // 00000000175C: 800A0608
	s_addc_u32 s11, s9, s7                                     // 000000001760: 820B0709
	s_load_b256 s[16:23], s[2:3], -0x30                        // 000000001764: F40C0401 F81FFFD0
	s_clause 0x7                                               // 00000000176C: BF850007
	s_load_b32 s12, s[10:11], null                             // 000000001770: F4000305 F8000000
	s_load_b32 s13, s[10:11], 0xb4                             // 000000001778: F4000345 F80000B4
	s_load_b32 s14, s[10:11], 0x168                            // 000000001780: F4000385 F8000168
	s_load_b32 s24, s[10:11], 0x21c                            // 000000001788: F4000605 F800021C
	s_load_b32 s25, s[10:11], 0x2d0                            // 000000001790: F4000645 F80002D0
	s_load_b32 s26, s[10:11], 0x384                            // 000000001798: F4000685 F8000384
	s_load_b32 s27, s[10:11], 0x438                            // 0000000017A0: F40006C5 F8000438
	s_load_b32 s28, s[10:11], 0x4ec                            // 0000000017A8: F4000705 F80004EC
	s_add_u32 s6, s6, 0x924                                    // 0000000017B0: 8006FF06 00000924
	s_addc_u32 s7, s7, 0                                       // 0000000017B8: 82078007
	s_waitcnt lgkmcnt(0)                                       // 0000000017BC: BF89FC07
	v_fmac_f32_e64 v0, s16, s12                                // 0000000017C0: D52B0000 00001810
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C8: BF870091
	v_fmac_f32_e64 v0, s17, s13                                // 0000000017CC: D52B0000 00001A11
	v_fmac_f32_e64 v0, s18, s14                                // 0000000017D4: D52B0000 00001C12
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017DC: BF870001
	v_fmac_f32_e64 v0, s19, s24                                // 0000000017E0: D52B0000 00003013
	s_clause 0x1                                               // 0000000017E8: BF850001
	s_load_b32 s12, s[2:3], null                               // 0000000017EC: F4000301 F8000000
	s_load_b128 s[16:19], s[2:3], -0x10                        // 0000000017F4: F4080401 F81FFFF0
	s_clause 0x1                                               // 0000000017FC: BF850001
	s_load_b32 s13, s[10:11], 0x5a0                            // 000000001800: F4000345 F80005A0
	s_load_b32 s14, s[10:11], 0x654                            // 000000001808: F4000385 F8000654
	s_add_u32 s2, s2, 52                                       // 000000001810: 8002B402
	s_addc_u32 s3, s3, 0                                       // 000000001814: 82038003
	v_fmac_f32_e64 v0, s20, s25                                // 000000001818: D52B0000 00003214
	s_load_b32 s20, s[10:11], 0x708                            // 000000001820: F4000505 F8000708
	s_cmpk_eq_i32 s6, 0x2db4                                   // 000000001828: B1862DB4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000182C: BF8700C1
	v_fmac_f32_e64 v0, s21, s26                                // 000000001830: D52B0000 00003415
	s_clause 0x1                                               // 000000001838: BF850001
	s_load_b32 s21, s[10:11], 0x7bc                            // 00000000183C: F4000545 F80007BC
	s_load_b32 s10, s[10:11], 0x870                            // 000000001844: F4000285 F8000870
	v_fmac_f32_e64 v0, s22, s27                                // 00000000184C: D52B0000 00003616
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001854: BF8700A1
	v_fmac_f32_e64 v0, s23, s28                                // 000000001858: D52B0000 00003817
	s_waitcnt lgkmcnt(0)                                       // 000000001860: BF89FC07
	v_fmac_f32_e64 v0, s16, s13                                // 000000001864: D52B0000 00001A10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000186C: BF870091
	v_fmac_f32_e64 v0, s17, s14                                // 000000001870: D52B0000 00001C11
	v_fmac_f32_e64 v0, s18, s20                                // 000000001878: D52B0000 00002812
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001880: BF870091
	v_fmac_f32_e64 v0, s19, s21                                // 000000001884: D52B0000 00002A13
	v_fmac_f32_e64 v0, s12, s10                                // 00000000188C: D52B0000 0000140C
	s_cbranch_scc0 65456                                       // 000000001894: BFA1FFB0 <r_450_45_65+0x58>
	s_mul_i32 s2, s15, 45                                      // 000000001898: 9602AD0F
	v_mov_b32_e32 v1, 0                                        // 00000000189C: 7E020280
	s_ashr_i32 s3, s2, 31                                      // 0000000018A0: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018A4: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000018A8: 84828202
	s_add_u32 s2, s4, s2                                       // 0000000018AC: 80020204
	s_addc_u32 s3, s5, s3                                      // 0000000018B0: 82030305
	s_add_u32 s0, s2, s0                                       // 0000000018B4: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000018B8: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000018BC: DC6A0000 00000001
	s_nop 0                                                    // 0000000018C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018C8: BFB60003
	s_endpgm                                                   // 0000000018CC: BFB00000
