
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_8775_3>:
	s_load_b256 s[0:7], s[0:1], null                           // 000000001700: F40C0000 F8000000
	s_mov_b32 s8, s15                                          // 000000001708: BE88000F
	s_mov_b32 s10, 0                                           // 00000000170C: BE8A0080
	s_cmp_gt_i32 s14, 0                                        // 000000001710: BF02800E
	s_mov_b32 s11, 0                                           // 000000001714: BE8B0080
	s_cbranch_scc0 34                                          // 000000001718: BFA10022 <E_8775_3+0xa4>
	s_cmp_lg_u32 s14, 1                                        // 00000000171C: BF07810E
	s_cbranch_scc0 42                                          // 000000001720: BFA1002A <E_8775_3+0xcc>
	s_cmp_lt_i32 s14, 2                                        // 000000001724: BF04820E
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_mov_b32 s2, 0                                            // 00000000172C: BE820080
	s_cbranch_scc1 7                                           // 000000001730: BFA20007 <E_8775_3+0x50>
	s_ashr_i32 s9, s8, 31                                      // 000000001734: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000173C: 84828208
	s_add_u32 s2, s6, s2                                       // 000000001740: 80020206
	s_addc_u32 s3, s7, s3                                      // 000000001744: 82030307
	s_load_b32 s2, s[2:3], null                                // 000000001748: F4000081 F8000000
	s_mul_i32 s4, s8, 3                                        // 000000001750: 96048308
	s_waitcnt lgkmcnt(0)                                       // 000000001754: BF89FC07
	v_add_f32_e64 v0, s10, s11                                 // 000000001758: D5030000 0000160A
	s_ashr_i32 s5, s4, 31                                      // 000000001760: 86059F04
	v_mov_b32_e32 v1, 0                                        // 000000001764: 7E020280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001768: 84848204
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704C2
	v_add_f32_e32 v0, s2, v0                                   // 000000001770: 06000002
	s_add_u32 s3, s0, s4                                       // 000000001774: 80030400
	s_addc_u32 s4, s1, s5                                      // 000000001778: 82040501
	s_ashr_i32 s15, s14, 31                                    // 00000000177C: 860F9F0E
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001780: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_add_u32 s0, s3, s0                                       // 000000001788: 80000003
	s_addc_u32 s1, s4, s1                                      // 00000000178C: 82010104
	global_store_b32 v1, v0, s[0:1]                            // 000000001790: DC6A0000 00000001
	s_nop 0                                                    // 000000001798: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000179C: BFB60003
	s_endpgm                                                   // 0000000017A0: BFB00000
	s_ashr_i32 s9, s8, 31                                      // 0000000017A4: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A8: BF870009
	s_lshl_b64 s[12:13], s[8:9], 2                             // 0000000017AC: 848C8208
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	s_add_u32 s2, s2, s12                                      // 0000000017B4: 80020C02
	s_addc_u32 s3, s3, s13                                     // 0000000017B8: 82030D03
	s_load_b32 s10, s[2:3], null                               // 0000000017BC: F4000281 F8000000
	s_cmp_lg_u32 s14, 1                                        // 0000000017C4: BF07810E
	s_cbranch_scc1 65494                                       // 0000000017C8: BFA2FFD6 <E_8775_3+0x24>
	s_ashr_i32 s9, s8, 31                                      // 0000000017CC: 86099F08
	s_waitcnt lgkmcnt(0)                                       // 0000000017D0: BF89FC07
	s_lshl_b64 s[2:3], s[8:9], 2                               // 0000000017D4: 84828208
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017D8: BF870009
	s_add_u32 s2, s4, s2                                       // 0000000017DC: 80020204
	s_addc_u32 s3, s5, s3                                      // 0000000017E0: 82030305
	s_load_b32 s11, s[2:3], null                               // 0000000017E4: F40002C1 F8000000
	s_cmp_lt_i32 s14, 2                                        // 0000000017EC: BF04820E
	s_mov_b32 s2, 0                                            // 0000000017F0: BE820080
	s_cbranch_scc0 65487                                       // 0000000017F4: BFA1FFCF <E_8775_3+0x34>
	s_branch 65493                                             // 0000000017F8: BFA0FFD5 <E_8775_3+0x50>
