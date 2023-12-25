
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_106_24_5_5>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 0xc08                                   // 000000001708: 9606FF0F 00000C08
	s_mul_i32 s8, s14, 28                                      // 000000001710: 96089C0E
	s_ashr_i32 s7, s6, 31                                      // 000000001714: 86079F06
	s_mov_b32 s4, s13                                          // 000000001718: BE84000D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000171C: 84868206
	v_mov_b32_e32 v1, 0                                        // 000000001720: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001728: 80050602
	s_addc_u32 s6, s3, s7                                      // 00000000172C: 82060703
	s_ashr_i32 s9, s8, 31                                      // 000000001730: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001738: 84828208
	s_add_u32 s7, s5, s2                                       // 00000000173C: 80070205
	s_addc_u32 s6, s6, s3                                      // 000000001740: 82060306
	s_ashr_i32 s5, s13, 31                                     // 000000001744: 86059F0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001748: BF870499
	s_lshl_b64 s[2:3], s[4:5], 2                               // 00000000174C: 84828204
	s_add_u32 s12, s7, s2                                      // 000000001750: 800C0207
	s_addc_u32 s13, s6, s3                                     // 000000001754: 820D0306
	s_clause 0x4                                               // 000000001758: BF850004
	s_load_b128 s[4:7], s[12:13], null                         // 00000000175C: F4080106 F8000000
	s_load_b32 s20, s[12:13], 0x10                             // 000000001764: F4000506 F8000010
	s_load_b128 s[8:11], s[12:13], 0x70                        // 00000000176C: F4080206 F8000070
	s_load_b32 s21, s[12:13], 0x80                             // 000000001774: F4000546 F8000080
	s_load_b128 s[16:19], s[12:13], 0xe0                       // 00000000177C: F4080406 F80000E0
	s_waitcnt lgkmcnt(0)                                       // 000000001784: BF89FC07
	v_max_f32_e64 v0, s4, s4                                   // 000000001788: D5100000 00000804
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001790: BF870091
	v_max_f32_e32 v0, 0xff800000, v0                           // 000000001794: 200000FF FF800000
	v_max3_f32 v0, s6, s5, v0                                  // 00000000179C: D61C0000 04000A06
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017A4: BF8700C1
	v_max3_f32 v0, s20, s7, v0                                 // 0000000017A8: D61C0000 04000E14
	s_clause 0x1                                               // 0000000017B0: BF850001
	s_load_b32 s20, s[12:13], 0xf0                             // 0000000017B4: F4000506 F80000F0
	s_load_b128 s[4:7], s[12:13], 0x150                        // 0000000017BC: F4080106 F8000150
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017C4: D61C0000 04001009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017CC: BF870091
	v_max3_f32 v0, s11, s10, v0                                // 0000000017D0: D61C0000 0400140B
	v_max3_f32 v0, s16, s21, v0                                // 0000000017D8: D61C0000 04002A10
	s_clause 0x2                                               // 0000000017E0: BF850002
	s_load_b32 s16, s[12:13], 0x160                            // 0000000017E4: F4000406 F8000160
	s_load_b128 s[8:11], s[12:13], 0x1c0                       // 0000000017EC: F4080206 F80001C0
	s_load_b32 s12, s[12:13], 0x1d0                            // 0000000017F4: F4000306 F80001D0
	v_max3_f32 v0, s18, s17, v0                                // 0000000017FC: D61C0000 04002212
	s_waitcnt lgkmcnt(0)                                       // 000000001804: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001808: BF870091
	v_max3_f32 v0, s20, s19, v0                                // 00000000180C: D61C0000 04002614
	v_max3_f32 v0, s5, s4, v0                                  // 000000001814: D61C0000 04000805
	s_mul_i32 s4, s15, 0x9f0                                   // 00000000181C: 9604FF0F 000009F0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001824: BF870099
	s_ashr_i32 s5, s4, 31                                      // 000000001828: 86059F04
	v_max3_f32 v0, s7, s6, v0                                  // 00000000182C: D61C0000 04000C07
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001834: 84848204
	s_mul_i32 s6, s14, 24                                      // 000000001838: 9606980E
	s_add_u32 s4, s0, s4                                       // 00000000183C: 80040400
	s_addc_u32 s5, s1, s5                                      // 000000001840: 82050501
	v_max3_f32 v0, s8, s16, v0                                 // 000000001844: D61C0000 04002008
	s_ashr_i32 s7, s6, 31                                      // 00000000184C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001850: BF870099
	s_lshl_b64 s[0:1], s[6:7], 2                               // 000000001854: 84808206
	v_max3_f32 v0, s10, s9, v0                                 // 000000001858: D61C0000 0400120A
	s_add_u32 s0, s4, s0                                       // 000000001860: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001864: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001868: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000186C: 82010301
	v_max3_f32 v0, s12, s11, v0                                // 000000001870: D61C0000 0400160C
	global_store_b32 v1, v0, s[0:1]                            // 000000001878: DC6A0000 00000001
	s_nop 0                                                    // 000000001880: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001884: BFB60003
	s_endpgm                                                   // 000000001888: BFB00000
