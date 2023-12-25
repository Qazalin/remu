
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_55_14_2_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 0xc24                                   // 000000001708: 9606FF0F 00000C24
	s_mul_i32 s8, s14, 56                                      // 000000001710: 9608B80E
	s_ashr_i32 s7, s6, 31                                      // 000000001714: 86079F06
	s_mov_b32 s4, s13                                          // 000000001718: BE84000D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000171C: 84868206
	v_mov_b32_e32 v1, 0                                        // 000000001720: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001728: 80050602
	s_addc_u32 s6, s3, s7                                      // 00000000172C: 82060703
	s_ashr_i32 s9, s8, 31                                      // 000000001730: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001734: BF8704D9
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001738: 84828208
	s_mul_i32 s8, s15, 0x302                                   // 00000000173C: 9608FF0F 00000302
	s_add_u32 s5, s5, s2                                       // 000000001744: 80050205
	s_addc_u32 s6, s6, s3                                      // 000000001748: 82060306
	s_lshl_b32 s2, s13, 1                                      // 00000000174C: 8402810D
	s_ashr_i32 s3, s2, 31                                      // 000000001750: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001754: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001758: 84828202
	s_add_u32 s2, s5, s2                                       // 00000000175C: 80020205
	s_addc_u32 s3, s6, s3                                      // 000000001760: 82030306
	s_clause 0x1                                               // 000000001764: BF850001
	s_load_b64 s[6:7], s[2:3], null                            // 000000001768: F4040181 F8000000
	s_load_b64 s[2:3], s[2:3], 0x70                            // 000000001770: F4040081 F8000070
	s_ashr_i32 s9, s8, 31                                      // 000000001778: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000177C: BF870499
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001780: 84888208
	s_add_u32 s5, s0, s8                                       // 000000001784: 80050800
	s_addc_u32 s8, s1, s9                                      // 000000001788: 82080901
	s_waitcnt lgkmcnt(0)                                       // 00000000178C: BF89FC07
	v_add_f32_e64 v0, s6, 0                                    // 000000001790: D5030000 00010006
	s_mul_i32 s6, s14, 14                                      // 000000001798: 96068E0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000179C: BF8704A1
	v_add_f32_e32 v0, s7, v0                                   // 0000000017A0: 06000007
	s_ashr_i32 s7, s6, 31                                      // 0000000017A4: 86079F06
	s_lshl_b64 s[0:1], s[6:7], 2                               // 0000000017A8: 84808206
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017AC: BF8700A1
	v_add_f32_e32 v0, s2, v0                                   // 0000000017B0: 06000002
	s_add_u32 s2, s5, s0                                       // 0000000017B4: 80020005
	v_add_f32_e32 v0, s3, v0                                   // 0000000017B8: 06000003
	s_addc_u32 s3, s8, s1                                      // 0000000017BC: 82030108
	s_ashr_i32 s5, s13, 31                                     // 0000000017C0: 86059F0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870099
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000017C8: 84808204
	v_mul_f32_e32 v0, 0x3e800000, v0                           // 0000000017CC: 100000FF 3E800000
	s_add_u32 s0, s2, s0                                       // 0000000017D4: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017D8: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000017DC: DC6A0000 00000001
	s_nop 0                                                    // 0000000017E4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017E8: BFB60003
	s_endpgm                                                   // 0000000017EC: BFB00000
