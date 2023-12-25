
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_37_14_2_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s14, 0x54                                    // 000000001708: 9606FF0E 00000054
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v1, 0      // 000000001710: CA1000FF 00000080 FF800000
	s_ashr_i32 s7, s6, 31                                      // 00000000171C: 86079F06
	s_mul_i32 s10, s14, 14                                     // 000000001720: 960A8E0E
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001724: 84868206
	s_mov_b32 s4, s13                                          // 000000001728: BE84000D
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001730: 80050602
	s_addc_u32 s7, s3, s7                                      // 000000001734: 82070703
	s_lshl_b32 s2, s13, 1                                      // 000000001738: 8402810D
	s_mul_i32 s6, s15, 0xc08                                   // 00000000173C: 9606FF0F 00000C08
	s_ashr_i32 s3, s2, 31                                      // 000000001744: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001748: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000174C: 84828202
	s_add_u32 s5, s5, s2                                       // 000000001750: 80050205
	s_addc_u32 s8, s7, s3                                      // 000000001754: 82080307
	s_ashr_i32 s7, s6, 31                                      // 000000001758: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000175C: BF870499
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001760: 84828206
	s_add_u32 s2, s5, s2                                       // 000000001764: 80020205
	s_addc_u32 s3, s8, s3                                      // 000000001768: 82030308
	s_clause 0x1                                               // 00000000176C: BF850001
	s_load_b64 s[6:7], s[2:3], null                            // 000000001770: F4040181 F8000000
	s_load_b64 s[2:3], s[2:3], 0x70                            // 000000001778: F4040081 F8000070
	s_mul_i32 s8, s15, 0x206                                   // 000000001780: 9608FF0F 00000206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001788: BF870499
	s_ashr_i32 s9, s8, 31                                      // 00000000178C: 86099F08
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001790: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001794: BF8704B9
	s_add_u32 s5, s0, s8                                       // 000000001798: 80050800
	s_addc_u32 s8, s1, s9                                      // 00000000179C: 82080901
	s_ashr_i32 s11, s10, 31                                    // 0000000017A0: 860B9F0A
	s_lshl_b64 s[0:1], s[10:11], 2                             // 0000000017A4: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000017A8: BF8704D9
	s_add_u32 s9, s5, s0                                       // 0000000017AC: 80090005
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_max3_f32 v0, s7, s6, v0                                  // 0000000017B4: D61C0000 04000C07
	s_addc_u32 s6, s8, s1                                      // 0000000017BC: 82060108
	s_ashr_i32 s5, s13, 31                                     // 0000000017C0: 86059F0D
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000017C4: 84808204
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017C8: BF870001
	v_max3_f32 v0, s3, s2, v0                                  // 0000000017CC: D61C0000 04000403
	s_add_u32 s0, s9, s0                                       // 0000000017D4: 80000009
	s_addc_u32 s1, s6, s1                                      // 0000000017D8: 82010106
	global_store_b32 v1, v0, s[0:1]                            // 0000000017DC: DC6A0000 00000001
	s_nop 0                                                    // 0000000017E4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017E8: BFB60003
	s_endpgm                                                   // 0000000017EC: BFB00000
