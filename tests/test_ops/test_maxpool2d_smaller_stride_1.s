
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_36_12_5_5>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 0xc08                                   // 000000001708: 9604FF0F 00000C08
	s_mul_i32 s6, s14, 0x54                                    // 000000001710: 9606FF0E 00000054
	s_ashr_i32 s5, s4, 31                                      // 000000001718: 86059F04
	s_mov_b32 s12, s13                                         // 00000000171C: BE8C000D
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001720: 84848204
	v_mov_b32_e32 v1, 0                                        // 000000001724: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s4, s2, s4                                       // 00000000172C: 80040402
	s_addc_u32 s5, s3, s5                                      // 000000001730: 82050503
	s_ashr_i32 s7, s6, 31                                      // 000000001734: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_lshl_b64 s[2:3], s[6:7], 2                               // 00000000173C: 84828206
	s_add_u32 s4, s4, s2                                       // 000000001740: 80040204
	s_addc_u32 s5, s5, s3                                      // 000000001744: 82050305
	s_lshl_b32 s2, s13, 1                                      // 000000001748: 8402810D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_ashr_i32 s3, s2, 31                                      // 000000001750: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001754: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001758: BF870009
	s_add_u32 s2, s4, s2                                       // 00000000175C: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001760: 82030305
	s_clause 0x4                                               // 000000001764: BF850004
	s_load_b128 s[4:7], s[2:3], null                           // 000000001768: F4080101 F8000000
	s_load_b32 s13, s[2:3], 0x10                               // 000000001770: F4000341 F8000010
	s_load_b128 s[8:11], s[2:3], 0x70                          // 000000001778: F4080201 F8000070
	s_load_b32 s20, s[2:3], 0x80                               // 000000001780: F4000501 F8000080
	s_load_b128 s[16:19], s[2:3], 0xe0                         // 000000001788: F4080401 F80000E0
	s_waitcnt lgkmcnt(0)                                       // 000000001790: BF89FC07
	v_max_f32_e64 v0, s4, s4                                   // 000000001794: D5100000 00000804
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000179C: BF870091
	v_max_f32_e32 v0, 0xff800000, v0                           // 0000000017A0: 200000FF FF800000
	v_max3_f32 v0, s6, s5, v0                                  // 0000000017A8: D61C0000 04000A06
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017B0: BF8700C1
	v_max3_f32 v0, s13, s7, v0                                 // 0000000017B4: D61C0000 04000E0D
	s_clause 0x1                                               // 0000000017BC: BF850001
	s_load_b32 s13, s[2:3], 0xf0                               // 0000000017C0: F4000341 F80000F0
	s_load_b128 s[4:7], s[2:3], 0x150                          // 0000000017C8: F4080101 F8000150
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017D0: D61C0000 04001009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870091
	v_max3_f32 v0, s11, s10, v0                                // 0000000017DC: D61C0000 0400140B
	v_max3_f32 v0, s16, s20, v0                                // 0000000017E4: D61C0000 04002810
	s_clause 0x1                                               // 0000000017EC: BF850001
	s_load_b32 s16, s[2:3], 0x160                              // 0000000017F0: F4000401 F8000160
	s_load_b128 s[8:11], s[2:3], 0x1c0                         // 0000000017F8: F4080201 F80001C0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001800: BF8704B1
	v_max3_f32 v0, s18, s17, v0                                // 000000001804: D61C0000 04002212
	s_load_b32 s17, s[2:3], 0x1d0                              // 00000000180C: F4000441 F80001D0
	s_mul_i32 s2, s15, 0x1b0                                   // 000000001814: 9602FF0F 000001B0
	s_ashr_i32 s3, s2, 31                                      // 00000000181C: 86039F02
	s_waitcnt lgkmcnt(0)                                       // 000000001820: BF89FC07
	v_max3_f32 v0, s13, s19, v0                                // 000000001824: D61C0000 0400260D
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000182C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001830: BF8700A9
	s_add_u32 s2, s0, s2                                       // 000000001834: 80020200
	s_addc_u32 s3, s1, s3                                      // 000000001838: 82030301
	v_max3_f32 v0, s5, s4, v0                                  // 00000000183C: D61C0000 04000805
	s_mul_i32 s4, s14, 12                                      // 000000001844: 96048C0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001848: BF870099
	s_ashr_i32 s5, s4, 31                                      // 00000000184C: 86059F04
	v_max3_f32 v0, s7, s6, v0                                  // 000000001850: D61C0000 04000C07
	s_lshl_b64 s[0:1], s[4:5], 2                               // 000000001858: 84808204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000185C: BF8700A9
	s_add_u32 s2, s2, s0                                       // 000000001860: 80020002
	s_addc_u32 s3, s3, s1                                      // 000000001864: 82030103
	v_max3_f32 v0, s8, s16, v0                                 // 000000001868: D61C0000 04002008
	s_ashr_i32 s13, s12, 31                                    // 000000001870: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001874: BF870099
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001878: 8480820C
	v_max3_f32 v0, s10, s9, v0                                 // 00000000187C: D61C0000 0400120A
	s_add_u32 s0, s2, s0                                       // 000000001884: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001888: 82010103
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000188C: BF870001
	v_max3_f32 v0, s17, s11, v0                                // 000000001890: D61C0000 04001611
	global_store_b32 v1, v0, s[0:1]                            // 000000001898: DC6A0000 00000001
	s_nop 0                                                    // 0000000018A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018A4: BFB60003
	s_endpgm                                                   // 0000000018A8: BFB00000
