
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_21_4_5_5>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 0xc08                                   // 000000001708: 9606FF0F 00000C08
	s_mul_i32 s8, s14, 0x8c                                    // 000000001710: 9608FF0E 0000008C
	s_ashr_i32 s7, s6, 31                                      // 000000001718: 86079F06
	s_mov_b32 s4, s13                                          // 00000000171C: BE84000D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001720: 84868206
	v_mov_b32_e32 v1, 0                                        // 000000001724: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s5, s2, s6                                       // 00000000172C: 80050602
	s_addc_u32 s7, s3, s7                                      // 000000001730: 82070703
	s_ashr_i32 s9, s8, 31                                      // 000000001734: 86099F08
	s_mul_i32 s6, s13, 5                                       // 000000001738: 9606850D
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000173C: 84828208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s5, s5, s2                                       // 000000001744: 80050205
	s_addc_u32 s8, s7, s3                                      // 000000001748: 82080307
	s_ashr_i32 s7, s6, 31                                      // 00000000174C: 86079F06
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001750: 84828206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s2, s5, s2                                       // 000000001758: 80020205
	s_addc_u32 s3, s8, s3                                      // 00000000175C: 82030308
	s_clause 0x7                                               // 000000001760: BF850007
	s_load_b32 s5, s[2:3], null                                // 000000001764: F4000141 F8000000
	s_load_b32 s6, s[2:3], 0xc                                 // 00000000176C: F4000181 F800000C
	s_load_b32 s7, s[2:3], 0x18                                // 000000001774: F40001C1 F8000018
	s_load_b32 s8, s[2:3], 0x24                                // 00000000177C: F4000201 F8000024
	s_load_b32 s9, s[2:3], 0x30                                // 000000001784: F4000241 F8000030
	s_load_b32 s10, s[2:3], 0xe0                               // 00000000178C: F4000281 F80000E0
	s_load_b32 s11, s[2:3], 0xec                               // 000000001794: F40002C1 F80000EC
	s_load_b32 s12, s[2:3], 0xf8                               // 00000000179C: F4000301 F80000F8
	s_waitcnt lgkmcnt(0)                                       // 0000000017A4: BF89FC07
	v_max_f32_e64 v0, s5, s5                                   // 0000000017A8: D5100000 00000A05
	s_load_b32 s5, s[2:3], 0x104                               // 0000000017B0: F4000141 F8000104
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B8: BF870091
	v_max_f32_e32 v0, 0xff800000, v0                           // 0000000017BC: 200000FF FF800000
	v_max3_f32 v0, s7, s6, v0                                  // 0000000017C4: D61C0000 04000C07
	s_clause 0x1                                               // 0000000017CC: BF850001
	s_load_b32 s6, s[2:3], 0x110                               // 0000000017D0: F4000181 F8000110
	s_load_b32 s7, s[2:3], 0x1c0                               // 0000000017D8: F40001C1 F80001C0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017E0: BF8700C1
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017E4: D61C0000 04001009
	s_clause 0x1                                               // 0000000017EC: BF850001
	s_load_b32 s8, s[2:3], 0x1cc                               // 0000000017F0: F4000201 F80001CC
	s_load_b32 s9, s[2:3], 0x1d8                               // 0000000017F8: F4000241 F80001D8
	v_max3_f32 v0, s11, s10, v0                                // 000000001800: D61C0000 0400140B
	s_clause 0x2                                               // 000000001808: BF850002
	s_load_b32 s10, s[2:3], 0x1e4                              // 00000000180C: F4000281 F80001E4
	s_load_b32 s11, s[2:3], 0x1f0                              // 000000001814: F40002C1 F80001F0
	s_load_b32 s13, s[2:3], 0x2a0                              // 00000000181C: F4000341 F80002A0
	s_waitcnt lgkmcnt(0)                                       // 000000001824: BF89FC07
	v_max3_f32 v0, s5, s12, v0                                 // 000000001828: D61C0000 04001805
	s_clause 0x2                                               // 000000001830: BF850002
	s_load_b32 s5, s[2:3], 0x2ac                               // 000000001834: F4000141 F80002AC
	s_load_b32 s12, s[2:3], 0x2b8                              // 00000000183C: F4000301 F80002B8
	s_load_b32 s16, s[2:3], 0x2c4                              // 000000001844: F4000401 F80002C4
	v_max3_f32 v0, s7, s6, v0                                  // 00000000184C: D61C0000 04000C07
	s_mul_i32 s6, s15, 0x54                                    // 000000001854: 9606FF0F 00000054
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000185C: BF870099
	s_ashr_i32 s7, s6, 31                                      // 000000001860: 86079F06
	v_max3_f32 v0, s9, s8, v0                                  // 000000001864: D61C0000 04001009
	s_clause 0x1                                               // 00000000186C: BF850001
	s_load_b32 s8, s[2:3], 0x2d0                               // 000000001870: F4000201 F80002D0
	s_load_b32 s9, s[2:3], 0x380                               // 000000001878: F4000241 F8000380
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001880: BF870001
	v_max3_f32 v0, s11, s10, v0                                // 000000001884: D61C0000 0400140B
	s_clause 0x3                                               // 00000000188C: BF850003
	s_load_b32 s10, s[2:3], 0x38c                              // 000000001890: F4000281 F800038C
	s_load_b32 s11, s[2:3], 0x398                              // 000000001898: F40002C1 F8000398
	s_load_b32 s15, s[2:3], 0x3a4                              // 0000000018A0: F40003C1 F80003A4
	s_load_b32 s17, s[2:3], 0x3b0                              // 0000000018A8: F4000441 F80003B0
	s_lshl_b64 s[2:3], s[6:7], 2                               // 0000000018B0: 84828206
	s_waitcnt lgkmcnt(0)                                       // 0000000018B4: BF89FC07
	v_max3_f32 v0, s5, s13, v0                                 // 0000000018B8: D61C0000 04001A05
	s_add_u32 s2, s0, s2                                       // 0000000018C0: 80020200
	s_addc_u32 s3, s1, s3                                      // 0000000018C4: 82030301
	s_lshl_b32 s0, s14, 2                                      // 0000000018C8: 8400820E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000018CC: BF8704A1
	v_max3_f32 v0, s16, s12, v0                                // 0000000018D0: D61C0000 04001810
	s_ashr_i32 s1, s0, 31                                      // 0000000018D8: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000018DC: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018E0: BF870099
	s_add_u32 s2, s2, s0                                       // 0000000018E4: 80020002
	v_max3_f32 v0, s9, s8, v0                                  // 0000000018E8: D61C0000 04001009
	s_addc_u32 s3, s3, s1                                      // 0000000018F0: 82030103
	s_ashr_i32 s5, s4, 31                                      // 0000000018F4: 86059F04
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F8: BF870099
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000018FC: 84808204
	v_max3_f32 v0, s11, s10, v0                                // 000000001900: D61C0000 0400140B
	s_add_u32 s0, s2, s0                                       // 000000001908: 80000002
	s_addc_u32 s1, s3, s1                                      // 00000000190C: 82010103
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001910: BF870001
	v_max3_f32 v0, s17, s15, v0                                // 000000001914: D61C0000 04001E11
	global_store_b32 v1, v0, s[0:1]                            // 00000000191C: DC6A0000 00000001
	s_nop 0                                                    // 000000001924: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001928: BFB60003
	s_endpgm                                                   // 00000000192C: BFB00000
