
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_36_9_3_3>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 0xc08                                   // 000000001708: 9606FF0F 00000C08
	s_mul_i32 s8, s14, 0x54                                    // 000000001710: 9608FF0E 00000054
	s_ashr_i32 s7, s6, 31                                      // 000000001718: 86079F06
	s_mov_b32 s4, s13                                          // 00000000171C: BE84000D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001720: 84868206
	v_mov_b32_e32 v1, 0                                        // 000000001724: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s5, s2, s6                                       // 00000000172C: 80050602
	s_addc_u32 s7, s3, s7                                      // 000000001730: 82070703
	s_ashr_i32 s9, s8, 31                                      // 000000001734: 86099F08
	s_mul_i32 s6, s13, 3                                       // 000000001738: 9606830D
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000173C: 84828208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s5, s5, s2                                       // 000000001744: 80050205
	s_addc_u32 s8, s7, s3                                      // 000000001748: 82080307
	s_ashr_i32 s7, s6, 31                                      // 00000000174C: 86079F06
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001750: 84828206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s2, s5, s2                                       // 000000001758: 80020205
	s_addc_u32 s3, s8, s3                                      // 00000000175C: 82030308
	s_clause 0x5                                               // 000000001760: BF850005
	s_load_b64 s[6:7], s[2:3], null                            // 000000001764: F4040181 F8000000
	s_load_b32 s5, s[2:3], 0x8                                 // 00000000176C: F4000141 F8000008
	s_load_b64 s[8:9], s[2:3], 0x70                            // 000000001774: F4040201 F8000070
	s_load_b32 s12, s[2:3], 0x78                               // 00000000177C: F4000301 F8000078
	s_load_b64 s[10:11], s[2:3], 0xe0                          // 000000001784: F4040281 F80000E0
	s_load_b32 s13, s[2:3], 0xe8                               // 00000000178C: F4000341 F80000E8
	s_mul_i32 s2, s15, 0x144                                   // 000000001794: 9602FF0F 00000144
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000179C: BF870499
	s_ashr_i32 s3, s2, 31                                      // 0000000017A0: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017A4: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000017A8: BF8700D9
	s_add_u32 s2, s0, s2                                       // 0000000017AC: 80020200
	s_addc_u32 s3, s1, s3                                      // 0000000017B0: 82030301
	s_waitcnt lgkmcnt(0)                                       // 0000000017B4: BF89FC07
	v_max_f32_e64 v0, s6, s6                                   // 0000000017B8: D5100000 00000C06
	s_mul_i32 s6, s14, 9                                       // 0000000017C0: 9606890E
	v_max_f32_e32 v0, 0xff800000, v0                           // 0000000017C4: 200000FF FF800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017CC: BF8704A1
	v_max3_f32 v0, s5, s7, v0                                  // 0000000017D0: D61C0000 04000E05
	s_ashr_i32 s7, s6, 31                                      // 0000000017D8: 86079F06
	s_lshl_b64 s[0:1], s[6:7], 2                               // 0000000017DC: 84808206
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017E0: BF8700C1
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017E4: D61C0000 04001009
	s_add_u32 s2, s2, s0                                       // 0000000017EC: 80020002
	s_addc_u32 s3, s3, s1                                      // 0000000017F0: 82030103
	s_ashr_i32 s5, s4, 31                                      // 0000000017F4: 86059F04
	v_max3_f32 v0, s10, s12, v0                                // 0000000017F8: D61C0000 0400180A
	s_lshl_b64 s[0:1], s[4:5], 2                               // 000000001800: 84808204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001804: BF8700A9
	s_add_u32 s0, s2, s0                                       // 000000001808: 80000002
	s_addc_u32 s1, s3, s1                                      // 00000000180C: 82010103
	v_max3_f32 v0, s13, s11, v0                                // 000000001810: D61C0000 0400160D
	global_store_b32 v1, v0, s[0:1]                            // 000000001818: DC6A0000 00000001
	s_nop 0                                                    // 000000001820: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001824: BFB60003
	s_endpgm                                                   // 000000001828: BFB00000
