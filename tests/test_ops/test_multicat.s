
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_45_195>:
	s_load_b256 s[0:7], s[0:1], null                           // 000000001700: F40C0000 F8000000
	s_mul_i32 s8, s15, 0x41                                    // 000000001708: 9608FF0F 00000041
	s_mov_b32 s10, 0                                           // 000000001710: BE8A0080
	s_add_i32 s8, s8, s14                                      // 000000001714: 81080E08
	s_cmp_gt_i32 s14, 64                                       // 000000001718: BF02C00E
	s_mov_b32 s11, 0                                           // 00000000171C: BE8B0080
	s_cbranch_scc0 38                                          // 000000001720: BFA10026 <E_45_195+0xbc>
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_i32 s2, s14, 0xffffffbf                              // 000000001728: 8102FF0E FFFFFFBF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001730: BF870009
	s_cmp_gt_u32 s2, 64                                        // 000000001734: BF08C002
	s_cbranch_scc0 45                                          // 000000001738: BFA1002D <E_45_195+0xf0>
	s_cmpk_lt_i32 s14, 0x82                                    // 00000000173C: B38E0082
	s_mov_b32 s2, 0                                            // 000000001740: BE820080
	s_cbranch_scc1 7                                           // 000000001744: BFA20007 <E_45_195+0x64>
	s_ashr_i32 s9, s8, 31                                      // 000000001748: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001750: 84828208
	s_add_u32 s2, s6, s2                                       // 000000001754: 80020206
	s_addc_u32 s3, s7, s3                                      // 000000001758: 82030307
	s_load_b32 s2, s[2:3], -0x208                              // 00000000175C: F4000081 F81FFDF8
	s_mul_i32 s4, s15, 0xc3                                    // 000000001764: 9604FF0F 000000C3
	s_waitcnt lgkmcnt(0)                                       // 00000000176C: BF89FC07
	v_add_f32_e64 v0, s11, s10                                 // 000000001770: D5030000 0000140B
	s_ashr_i32 s5, s4, 31                                      // 000000001778: 86059F04
	v_mov_b32_e32 v1, 0                                        // 00000000177C: 7E020280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001780: 84848204
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001784: BF8704C2
	v_add_f32_e32 v0, s2, v0                                   // 000000001788: 06000002
	s_add_u32 s3, s0, s4                                       // 00000000178C: 80030400
	s_addc_u32 s4, s1, s5                                      // 000000001790: 82040501
	s_ashr_i32 s15, s14, 31                                    // 000000001794: 860F9F0E
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001798: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000179C: BF870009
	s_add_u32 s0, s3, s0                                       // 0000000017A0: 80000003
	s_addc_u32 s1, s4, s1                                      // 0000000017A4: 82010104
	global_store_b32 v1, v0, s[0:1]                            // 0000000017A8: DC6A0000 00000001
	s_nop 0                                                    // 0000000017B0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017B4: BFB60003
	s_endpgm                                                   // 0000000017B8: BFB00000
	s_ashr_i32 s9, s8, 31                                      // 0000000017BC: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017C0: BF870009
	s_lshl_b64 s[12:13], s[8:9], 2                             // 0000000017C4: 848C8208
	s_waitcnt lgkmcnt(0)                                       // 0000000017C8: BF89FC07
	s_add_u32 s2, s2, s12                                      // 0000000017CC: 80020C02
	s_addc_u32 s3, s3, s13                                     // 0000000017D0: 82030D03
	s_load_b32 s11, s[2:3], null                               // 0000000017D4: F40002C1 F8000000
	s_add_i32 s2, s14, 0xffffffbf                              // 0000000017DC: 8102FF0E FFFFFFBF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017E4: BF870009
	s_cmp_gt_u32 s2, 64                                        // 0000000017E8: BF08C002
	s_cbranch_scc1 65491                                       // 0000000017EC: BFA2FFD3 <E_45_195+0x3c>
	s_ashr_i32 s9, s8, 31                                      // 0000000017F0: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017F4: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 0000000017F8: 84828208
	s_add_u32 s2, s4, s2                                       // 0000000017FC: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001800: 82030305
	s_load_b32 s10, s[2:3], -0x104                             // 000000001804: F4000281 F81FFEFC
	s_cmpk_lt_i32 s14, 0x82                                    // 00000000180C: B38E0082
	s_mov_b32 s2, 0                                            // 000000001810: BE820080
	s_cbranch_scc0 65484                                       // 000000001814: BFA1FFCC <E_45_195+0x48>
	s_branch 65490                                             // 000000001818: BFA0FFD2 <E_45_195+0x64>
