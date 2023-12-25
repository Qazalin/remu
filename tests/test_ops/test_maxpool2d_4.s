
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_36_14_3_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 0xc08                                   // 000000001708: 9606FF0F 00000C08
	s_mul_i32 s8, s14, 0x54                                    // 000000001710: 9608FF0E 00000054
	s_ashr_i32 s7, s6, 31                                      // 000000001718: 86079F06
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v1, 0      // 00000000171C: CA1000FF 00000080 FF800000
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001728: 84868206
	s_mul_i32 s10, s15, 0x1f8                                  // 00000000172C: 960AFF0F 000001F8
	s_mul_i32 s12, s14, 14                                     // 000000001734: 960C8E0E
	s_mov_b32 s4, s13                                          // 000000001738: BE84000D
	s_waitcnt lgkmcnt(0)                                       // 00000000173C: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001740: 80050602
	s_addc_u32 s6, s3, s7                                      // 000000001744: 82060703
	s_ashr_i32 s9, s8, 31                                      // 000000001748: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001750: 84828208
	s_add_u32 s5, s5, s2                                       // 000000001754: 80050205
	s_addc_u32 s6, s6, s3                                      // 000000001758: 82060306
	s_lshl_b32 s2, s13, 1                                      // 00000000175C: 8402810D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001760: BF870499
	s_ashr_i32 s3, s2, 31                                      // 000000001764: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001768: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000176C: BF870009
	s_add_u32 s2, s5, s2                                       // 000000001770: 80020205
	s_addc_u32 s3, s6, s3                                      // 000000001774: 82030306
	s_clause 0x2                                               // 000000001778: BF850002
	s_load_b64 s[6:7], s[2:3], null                            // 00000000177C: F4040181 F8000000
	s_load_b64 s[8:9], s[2:3], 0x70                            // 000000001784: F4040201 F8000070
	s_load_b64 s[2:3], s[2:3], 0xe0                            // 00000000178C: F4040081 F80000E0
	s_ashr_i32 s11, s10, 31                                    // 000000001794: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001798: BF870499
	s_lshl_b64 s[10:11], s[10:11], 2                           // 00000000179C: 848A820A
	s_add_u32 s5, s0, s10                                      // 0000000017A0: 80050A00
	s_addc_u32 s10, s1, s11                                    // 0000000017A4: 820A0B01
	s_ashr_i32 s13, s12, 31                                    // 0000000017A8: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017AC: BF870009
	s_lshl_b64 s[0:1], s[12:13], 2                             // 0000000017B0: 8480820C
	s_waitcnt lgkmcnt(0)                                       // 0000000017B4: BF89FC07
	v_max3_f32 v0, s7, s6, v0                                  // 0000000017B8: D61C0000 04000C07
	s_add_u32 s6, s5, s0                                       // 0000000017C0: 80060005
	s_addc_u32 s7, s10, s1                                     // 0000000017C4: 8207010A
	s_ashr_i32 s5, s4, 31                                      // 0000000017C8: 86059F04
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017CC: BF8704A1
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017D0: D61C0000 04001009
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000017D8: 84808204
	s_add_u32 s0, s6, s0                                       // 0000000017DC: 80000006
	s_addc_u32 s1, s7, s1                                      // 0000000017E0: 82010107
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017E4: BF870001
	v_max3_f32 v0, s3, s2, v0                                  // 0000000017E8: D61C0000 04000403
	global_store_b32 v1, v0, s[0:1]                            // 0000000017F0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017F8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017FC: BFB60003
	s_endpgm                                                   // 000000001800: BFB00000
