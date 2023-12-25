
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_9_10_6>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_add_i32 s5, s13, -1                                      // 000000001708: 8105C10D
	s_mov_b32 s4, s13                                          // 00000000170C: BE84000D
	s_cmp_gt_u32 s5, 2                                         // 000000001710: BF088205
	s_cselect_b32 s5, -1, 0                                    // 000000001714: 980580C1
	s_add_i32 s6, s14, -3                                      // 000000001718: 8106C30E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000171C: BF8704A9
	s_cmp_gt_u32 s6, 2                                         // 000000001720: BF088206
	s_cselect_b32 s6, -1, 0                                    // 000000001724: 980680C1
	s_or_b32 s5, s5, s6                                        // 000000001728: 8C050605
	s_mov_b32 s6, 0                                            // 00000000172C: BE860080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001730: 8B6A057E
	s_cbranch_vccnz 18                                         // 000000001734: BFA40012 <E_9_10_6+0x80>
	s_mul_i32 s6, s15, 9                                       // 000000001738: 9606890F
	s_mul_i32 s8, s14, 3                                       // 00000000173C: 9608830E
	s_ashr_i32 s7, s6, 31                                      // 000000001740: 86079F06
	s_mov_b32 s9, 0                                            // 000000001744: BE890080
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001748: 84868206
	s_mov_b32 s5, s9                                           // 00000000174C: BE850009
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_add_u32 s6, s2, s6                                       // 000000001754: 80060602
	s_addc_u32 s7, s3, s7                                      // 000000001758: 82070703
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000175C: 84828208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001760: BF8704B9
	s_add_u32 s6, s6, s2                                       // 000000001764: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001768: 82070307
	s_lshl_b64 s[2:3], s[4:5], 2                               // 00000000176C: 84828204
	s_add_u32 s2, s6, s2                                       // 000000001770: 80020206
	s_addc_u32 s3, s7, s3                                      // 000000001774: 82030307
	s_load_b32 s6, s[2:3], -0x28                               // 000000001778: F4000181 F81FFFD8
	s_waitcnt lgkmcnt(0)                                       // 000000001780: BF89FC07
	s_mul_i32 s2, s15, 60                                      // 000000001784: 9602BC0F
	s_mul_i32 s8, s14, 6                                       // 000000001788: 9608860E
	s_ashr_i32 s3, s2, 31                                      // 00000000178C: 86039F02
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s6              // 000000001790: CA100080 00000006
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001798: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000179C: BF8704B9
	s_add_u32 s2, s0, s2                                       // 0000000017A0: 80020200
	s_addc_u32 s3, s1, s3                                      // 0000000017A4: 82030301
	s_ashr_i32 s9, s8, 31                                      // 0000000017A8: 86099F08
	s_lshl_b64 s[0:1], s[8:9], 2                               // 0000000017AC: 84808208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017B0: BF8704B9
	s_add_u32 s2, s2, s0                                       // 0000000017B4: 80020002
	s_addc_u32 s3, s3, s1                                      // 0000000017B8: 82030103
	s_ashr_i32 s5, s4, 31                                      // 0000000017BC: 86059F04
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000017C0: 84808204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017C4: BF870009
	s_add_u32 s0, s2, s0                                       // 0000000017C8: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017CC: 82010103
	global_store_b32 v0, v1, s[0:1]                            // 0000000017D0: DC6A0000 00000100
	s_nop 0                                                    // 0000000017D8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017DC: BFB60003
	s_endpgm                                                   // 0000000017E0: BFB00000
