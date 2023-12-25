
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_9_6_3_3_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s14, 7                                       // 000000001714: 9608870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001720: BF870009
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000172C: 80080806
	s_addc_u32 s7, s7, s9                                      // 000000001730: 82070907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_mul_i32 s6, s15, 18                                      // 000000001738: 9606920F
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000173C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s8, s8, s2                                       // 000000001744: 80080208
	s_addc_u32 s9, s7, s3                                      // 000000001748: 82090307
	s_ashr_i32 s7, s6, 31                                      // 00000000174C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001750: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s0, s0, s6                                       // 000000001758: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000175C: 82010701
	s_load_b512 s[16:31], s[0:1], null                         // 000000001760: F4100400 F8000000
	s_clause 0x3                                               // 000000001768: BF850003
	s_load_b64 s[6:7], s[8:9], null                            // 00000000176C: F4040184 F8000000
	s_load_b64 s[10:11], s[8:9], 0x1c                          // 000000001774: F4040284 F800001C
	s_load_b64 s[12:13], s[8:9], 0x38                          // 00000000177C: F4040304 F8000038
	s_load_b64 s[34:35], s[8:9], 0x134                         // 000000001784: F4040884 F8000134
	s_waitcnt lgkmcnt(0)                                       // 00000000178C: BF89FC07
	v_fma_f32 v0, s6, s16, 0                                   // 000000001790: D6130000 02002006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001798: BF8700A1
	v_fmac_f32_e64 v0, s7, s17                                 // 00000000179C: D52B0000 00002207
	s_load_b64 s[6:7], s[8:9], 0x150                           // 0000000017A4: F4040184 F8000150
	v_fmac_f32_e64 v0, s10, s18                                // 0000000017AC: D52B0000 0000240A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017B4: BF8700A1
	v_fmac_f32_e64 v0, s11, s19                                // 0000000017B8: D52B0000 0000260B
	s_load_b64 s[10:11], s[8:9], 0x16c                         // 0000000017C0: F4040284 F800016C
	v_fmac_f32_e64 v0, s12, s20                                // 0000000017C8: D52B0000 0000280C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017D0: BF8700C1
	v_fmac_f32_e64 v0, s13, s21                                // 0000000017D4: D52B0000 00002A0D
	s_clause 0x1                                               // 0000000017DC: BF850001
	s_load_b64 s[12:13], s[8:9], 0x268                         // 0000000017E0: F4040304 F8000268
	s_load_b64 s[16:17], s[8:9], 0x284                         // 0000000017E8: F4040404 F8000284
	v_fmac_f32_e64 v0, s34, s22                                // 0000000017F0: D52B0000 00002C22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017F8: BF8700A1
	v_fmac_f32_e64 v0, s35, s23                                // 0000000017FC: D52B0000 00002E23
	s_waitcnt lgkmcnt(0)                                       // 000000001804: BF89FC07
	v_fmac_f32_e64 v0, s6, s24                                 // 000000001808: D52B0000 00003006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001810: BF8704C1
	v_fmac_f32_e64 v0, s7, s25                                 // 000000001814: D52B0000 00003207
	s_load_b64 s[6:7], s[8:9], 0x2a0                           // 00000000181C: F4040184 F80002A0
	s_load_b64 s[0:1], s[0:1], 0x40                            // 000000001824: F4040000 F8000040
	s_mul_i32 s8, s15, 54                                      // 00000000182C: 9608B60F
	s_ashr_i32 s9, s8, 31                                      // 000000001830: 86099F08
	v_fmac_f32_e64 v0, s10, s26                                // 000000001834: D52B0000 0000340A
	s_lshl_b64 s[8:9], s[8:9], 2                               // 00000000183C: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001840: BF8700A9
	s_add_u32 s4, s4, s8                                       // 000000001844: 80040804
	s_addc_u32 s5, s5, s9                                      // 000000001848: 82050905
	v_fmac_f32_e64 v0, s11, s27                                // 00000000184C: D52B0000 0000360B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001854: BF870091
	v_fmac_f32_e64 v0, s12, s28                                // 000000001858: D52B0000 0000380C
	v_fmac_f32_e64 v0, s13, s29                                // 000000001860: D52B0000 00003A0D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001868: BF870091
	v_fmac_f32_e64 v0, s16, s30                                // 00000000186C: D52B0000 00003C10
	v_fmac_f32_e64 v0, s17, s31                                // 000000001874: D52B0000 00003E11
	s_waitcnt lgkmcnt(0)                                       // 00000000187C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001880: BF8700A1
	v_fmac_f32_e64 v0, s6, s0                                  // 000000001884: D52B0000 00000006
	s_mul_i32 s0, s14, 6                                       // 00000000188C: 9600860E
	v_fmac_f32_e64 v0, s7, s1                                  // 000000001890: D52B0000 00000207
	s_ashr_i32 s1, s0, 31                                      // 000000001898: 86019F00
	v_mov_b32_e32 v1, 0                                        // 00000000189C: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000018A0: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000018A4: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000018A8: 20000080
	s_add_u32 s0, s4, s0                                       // 0000000018AC: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000018B0: 82010105
	s_add_u32 s0, s0, s2                                       // 0000000018B4: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000018B8: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000018BC: DC6A0000 00000001
	s_nop 0                                                    // 0000000018C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018C8: BFB60003
	s_endpgm                                                   // 0000000018CC: BFB00000
