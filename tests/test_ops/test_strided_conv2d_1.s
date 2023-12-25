
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_4_5_26_3_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_hi_i32 s1, s13, 0x4ec4ec4f                           // 000000001714: 9701FF0D 4EC4EC4F
	s_mul_i32 s0, s15, 0x39c                                   // 00000000171C: 9600FF0F 0000039C
	s_lshr_b32 s2, s1, 31                                      // 000000001724: 85029F01
	s_ashr_i32 s3, s1, 3                                       // 000000001728: 86038301
	s_ashr_i32 s1, s0, 31                                      // 00000000172C: 86019F00
	s_add_i32 s10, s3, s2                                      // 000000001730: 810A0203
	s_lshl_b64 s[2:3], s[0:1], 2                               // 000000001734: 84828200
	s_mul_i32 s0, s10, 26                                      // 000000001738: 96009A0A
	s_mul_i32 s10, s10, 56                                     // 00000000173C: 960AB80A
	s_sub_i32 s12, s13, s0                                     // 000000001740: 818C000D
	s_waitcnt lgkmcnt(0)                                       // 000000001744: BF89FC07
	s_add_u32 s1, s6, s2                                       // 000000001748: 80010206
	s_addc_u32 s6, s7, s3                                      // 00000000174C: 82060307
	s_ashr_i32 s11, s10, 31                                    // 000000001750: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001754: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001758: 8482820A
	s_add_u32 s1, s1, s2                                       // 00000000175C: 80010201
	s_addc_u32 s7, s6, s3                                      // 000000001760: 82070306
	s_ashr_i32 s13, s12, 31                                    // 000000001764: 860D9F0C
	s_mul_i32 s6, s14, 27                                      // 000000001768: 96069B0E
	s_lshl_b64 s[2:3], s[12:13], 2                             // 00000000176C: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001770: BF8704B9
	s_add_u32 s10, s1, s2                                      // 000000001774: 800A0201
	s_addc_u32 s11, s7, s3                                     // 000000001778: 820B0307
	s_ashr_i32 s7, s6, 31                                      // 00000000177C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001780: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_add_u32 s6, s8, s6                                       // 000000001788: 80060608
	s_addc_u32 s7, s9, s7                                      // 00000000178C: 82070709
	s_load_b512 s[16:31], s[6:7], null                         // 000000001790: F4100403 F8000000
	s_clause 0x3                                               // 000000001798: BF850003
	s_load_b64 s[8:9], s[10:11], null                          // 00000000179C: F4040205 F8000000
	s_load_b32 s1, s[10:11], 0x8                               // 0000000017A4: F4000045 F8000008
	s_load_b64 s[12:13], s[10:11], 0x70                        // 0000000017AC: F4040305 F8000070
	s_load_b32 s33, s[10:11], 0x78                             // 0000000017B4: F4000845 F8000078
	s_waitcnt lgkmcnt(0)                                       // 0000000017BC: BF89FC07
	v_fma_f32 v0, s8, s16, 0                                   // 0000000017C0: D6130000 02002008
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017C8: BF8700C1
	v_fmac_f32_e64 v0, s9, s17                                 // 0000000017CC: D52B0000 00002209
	s_clause 0x1                                               // 0000000017D4: BF850001
	s_load_b64 s[8:9], s[10:11], 0x4d0                         // 0000000017D8: F4040205 F80004D0
	s_load_b64 s[16:17], s[10:11], 0xe0                        // 0000000017E0: F4040405 F80000E0
	v_fmac_f32_e64 v0, s1, s18                                 // 0000000017E8: D52B0000 00002401
	s_load_b32 s1, s[10:11], 0xe8                              // 0000000017F0: F4000045 F80000E8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F8: BF870091
	v_fmac_f32_e64 v0, s12, s19                                // 0000000017FC: D52B0000 0000260C
	v_fmac_f32_e64 v0, s13, s20                                // 000000001804: D52B0000 0000280D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000180C: BF8700A1
	v_fmac_f32_e64 v0, s33, s21                                // 000000001810: D52B0000 00002A21
	s_waitcnt lgkmcnt(0)                                       // 000000001818: BF89FC07
	v_fmac_f32_e64 v0, s16, s22                                // 00000000181C: D52B0000 00002C10
	s_clause 0x2                                               // 000000001824: BF850002
	s_load_b32 s16, s[10:11], 0x4d8                            // 000000001828: F4000405 F80004D8
	s_load_b64 s[12:13], s[10:11], 0x5b0                       // 000000001830: F4040305 F80005B0
	s_load_b64 s[34:35], s[10:11], 0x540                       // 000000001838: F4040885 F8000540
	v_fmac_f32_e64 v0, s17, s23                                // 000000001840: D52B0000 00002E11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001848: BF8700A1
	v_fmac_f32_e64 v0, s1, s24                                 // 00000000184C: D52B0000 00003001
	s_load_b32 s1, s[10:11], 0x548                             // 000000001854: F4000045 F8000548
	v_fmac_f32_e64 v0, s8, s25                                 // 00000000185C: D52B0000 00003208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001864: BF8700B1
	v_fmac_f32_e64 v0, s9, s26                                 // 000000001868: D52B0000 00003409
	s_load_b32 s26, s[10:11], 0x5b8                            // 000000001870: F4000685 F80005B8
	s_waitcnt lgkmcnt(0)                                       // 000000001878: BF89FC07
	v_fmac_f32_e64 v0, s16, s27                                // 00000000187C: D52B0000 00003610
	s_load_b256 s[16:23], s[6:7], 0x40                         // 000000001884: F40C0403 F8000040
	s_clause 0x1                                               // 00000000188C: BF850001
	s_load_b64 s[8:9], s[10:11], 0xa10                         // 000000001890: F4040205 F8000A10
	s_load_b64 s[24:25], s[10:11], 0x9a0                       // 000000001898: F4040605 F80009A0
	v_fmac_f32_e64 v0, s34, s28                                // 0000000018A0: D52B0000 00003822
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018A8: BF870091
	v_fmac_f32_e64 v0, s35, s29                                // 0000000018AC: D52B0000 00003A23
	v_fmac_f32_e64 v0, s1, s30                                 // 0000000018B4: D52B0000 00003C01
	s_load_b32 s1, s[10:11], 0x9a8                             // 0000000018BC: F4000045 F80009A8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018C4: BF8700A1
	v_fmac_f32_e64 v0, s12, s31                                // 0000000018C8: D52B0000 00003E0C
	s_waitcnt lgkmcnt(0)                                       // 0000000018D0: BF89FC07
	v_fmac_f32_e64 v0, s13, s16                                // 0000000018D4: D52B0000 0000200D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018DC: BF870091
	v_fmac_f32_e64 v0, s26, s17                                // 0000000018E0: D52B0000 0000221A
	v_fmac_f32_e64 v0, s24, s18                                // 0000000018E8: D52B0000 00002418
	s_clause 0x1                                               // 0000000018F0: BF850001
	s_load_b32 s18, s[10:11], 0xa18                            // 0000000018F4: F4000485 F8000A18
	s_load_b64 s[12:13], s[10:11], 0xa80                       // 0000000018FC: F4040305 F8000A80
	s_load_b64 s[16:17], s[6:7], 0x60                          // 000000001904: F4040403 F8000060
	v_fmac_f32_e64 v0, s25, s19                                // 00000000190C: D52B0000 00002619
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001914: BF8704C1
	v_fmac_f32_e64 v0, s1, s20                                 // 000000001918: D52B0000 00002801
	s_load_b32 s1, s[10:11], 0xa88                             // 000000001920: F4000045 F8000A88
	s_load_b32 s10, s[6:7], 0x68                               // 000000001928: F4000283 F8000068
	s_mul_i32 s6, s15, 0x208                                   // 000000001930: 9606FF0F 00000208
	s_ashr_i32 s7, s6, 31                                      // 000000001938: 86079F06
	v_fmac_f32_e64 v0, s8, s21                                 // 00000000193C: D52B0000 00002A08
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001944: 84868206
	s_mul_i32 s8, s14, 0x82                                    // 000000001948: 9608FF0E 00000082
	s_add_u32 s6, s4, s6                                       // 000000001950: 80060604
	s_addc_u32 s7, s5, s7                                      // 000000001954: 82070705
	v_fmac_f32_e64 v0, s9, s22                                 // 000000001958: D52B0000 00002C09
	s_ashr_i32 s9, s8, 31                                      // 000000001960: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001964: BF8700A9
	s_lshl_b64 s[4:5], s[8:9], 2                               // 000000001968: 84848208
	s_waitcnt lgkmcnt(0)                                       // 00000000196C: BF89FC07
	v_fmac_f32_e64 v0, s18, s23                                // 000000001970: D52B0000 00002E12
	s_add_u32 s4, s6, s4                                       // 000000001978: 80040406
	s_addc_u32 s5, s7, s5                                      // 00000000197C: 82050507
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001980: BF870091
	v_fmac_f32_e64 v0, s12, s16                                // 000000001984: D52B0000 0000200C
	v_fmac_f32_e64 v0, s13, s17                                // 00000000198C: D52B0000 0000220D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001994: BF870141
	v_fmac_f32_e64 v0, s1, s10                                 // 000000001998: D52B0000 00001401
	s_ashr_i32 s1, s0, 31                                      // 0000000019A0: 86019F00
	v_mov_b32_e32 v1, 0                                        // 0000000019A4: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000019A8: 84808200
	v_max_f32_e32 v0, 0, v0                                    // 0000000019AC: 20000080
	s_add_u32 s0, s4, s0                                       // 0000000019B0: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000019B4: 82010105
	s_add_u32 s0, s0, s2                                       // 0000000019B8: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000019BC: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000019C0: DC6A0000 00000001
	s_nop 0                                                    // 0000000019C8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000019CC: BFB60003
	s_endpgm                                                   // 0000000019D0: BFB00000
