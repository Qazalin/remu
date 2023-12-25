
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_3_3_2_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s15, 3                                       // 000000001714: 9608830F
	s_mov_b32 s2, s15                                          // 000000001718: BE82000F
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_cmp_lt_i32 s15, 3                                        // 000000001720: BF04830F
	s_mov_b32 s10, 0                                           // 000000001724: BE8A0080
	s_cselect_b32 s12, -1, 0                                   // 000000001728: 980C80C1
	s_cmp_lt_i32 s14, 3                                        // 00000000172C: BF04830E
	s_cselect_b32 s3, -1, 0                                    // 000000001730: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_and_b32 s11, s12, s3                                     // 000000001738: 8B0B030C
	s_and_not1_b32 vcc_lo, exec_lo, s11                        // 00000000173C: 916A0B7E
	s_mov_b32 s11, 0                                           // 000000001740: BE8B0080
	s_cbranch_vccnz 11                                         // 000000001744: BFA4000B <r_3_3_2_2+0x74>
	s_lshl_b64 s[16:17], s[8:9], 2                             // 000000001748: 84908208
	s_ashr_i32 s15, s14, 31                                    // 00000000174C: 860F9F0E
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_add_u32 s11, s6, s16                                     // 000000001754: 800B1006
	s_addc_u32 s13, s7, s17                                    // 000000001758: 820D1107
	s_lshl_b64 s[16:17], s[14:15], 2                           // 00000000175C: 8490820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001760: BF870009
	s_add_u32 s16, s11, s16                                    // 000000001764: 8010100B
	s_addc_u32 s17, s13, s17                                   // 000000001768: 8211110D
	s_load_b32 s11, s[16:17], null                             // 00000000176C: F40002C8 F8000000
	s_cmp_lt_i32 s14, 2                                        // 000000001774: BF04820E
	s_cselect_b32 s13, -1, 0                                   // 000000001778: 980D80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000177C: BF870499
	s_and_b32 s12, s12, s13                                    // 000000001780: 8B0C0D0C
	s_and_not1_b32 vcc_lo, exec_lo, s12                        // 000000001784: 916A0C7E
	s_cbranch_vccnz 11                                         // 000000001788: BFA4000B <r_3_3_2_2+0xb8>
	s_lshl_b64 s[16:17], s[8:9], 2                             // 00000000178C: 84908208
	s_waitcnt lgkmcnt(0)                                       // 000000001790: BF89FC07
	s_add_u32 s10, s6, s16                                     // 000000001794: 800A1006
	s_addc_u32 s12, s7, s17                                    // 000000001798: 820C1107
	s_ashr_i32 s15, s14, 31                                    // 00000000179C: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017A0: BF870499
	s_lshl_b64 s[16:17], s[14:15], 2                           // 0000000017A4: 8490820E
	s_add_u32 s16, s10, s16                                    // 0000000017A8: 8010100A
	s_addc_u32 s17, s12, s17                                   // 0000000017AC: 8211110C
	s_load_b32 s10, s[16:17], 0x4                              // 0000000017B0: F4000288 F8000004
	s_cmp_lt_i32 s2, 2                                         // 0000000017B8: BF048202
	s_mov_b32 s12, 0                                           // 0000000017BC: BE8C0080
	s_cselect_b32 s2, -1, 0                                    // 0000000017C0: 980280C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017C4: BF870499
	s_and_b32 s3, s2, s3                                       // 0000000017C8: 8B030302
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 0000000017CC: 916A037E
	s_cbranch_vccnz 11                                         // 0000000017D0: BFA4000B <r_3_3_2_2+0x100>
	s_lshl_b64 s[16:17], s[8:9], 2                             // 0000000017D4: 84908208
	s_ashr_i32 s15, s14, 31                                    // 0000000017D8: 860F9F0E
	s_waitcnt lgkmcnt(0)                                       // 0000000017DC: BF89FC07
	s_add_u32 s3, s6, s16                                      // 0000000017E0: 80031006
	s_addc_u32 s12, s7, s17                                    // 0000000017E4: 820C1107
	s_lshl_b64 s[16:17], s[14:15], 2                           // 0000000017E8: 8490820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017EC: BF870009
	s_add_u32 s16, s3, s16                                     // 0000000017F0: 80101003
	s_addc_u32 s17, s12, s17                                   // 0000000017F4: 8211110C
	s_load_b32 s12, s[16:17], 0xc                              // 0000000017F8: F4000308 F800000C
	s_and_b32 s2, s2, s13                                      // 000000001800: 8B020D02
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001804: BF870009
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001808: 8B6A027E
	s_cbranch_vccnz 10                                         // 00000000180C: BFA4000A <r_3_3_2_2+0x138>
	s_ashr_i32 s15, s14, 31                                    // 000000001810: 860F9F0E
	s_mov_b32 s13, 0                                           // 000000001814: BE8D0080
	s_waitcnt lgkmcnt(0)                                       // 000000001818: BF89FC07
	s_clause 0x1                                               // 00000000181C: BF850001
	s_load_b64 s[2:3], s[0:1], null                            // 000000001820: F4040080 F8000000
	s_load_b32 s16, s[0:1], 0x8                                // 000000001828: F4000400 F8000008
	s_cbranch_execz 7                                          // 000000001830: BFA50007 <r_3_3_2_2+0x150>
	s_branch 17                                                // 000000001834: BFA00011 <r_3_3_2_2+0x17c>
	s_waitcnt lgkmcnt(0)                                       // 000000001838: BF89FC07
	s_clause 0x1                                               // 00000000183C: BF850001
	s_load_b64 s[2:3], s[0:1], null                            // 000000001840: F4040080 F8000000
	s_load_b32 s16, s[0:1], 0x8                                // 000000001848: F4000400 F8000008
	s_lshl_b64 s[18:19], s[8:9], 2                             // 000000001850: 84928208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001854: BF8704B9
	s_add_u32 s13, s6, s18                                     // 000000001858: 800D1206
	s_addc_u32 s17, s7, s19                                    // 00000000185C: 82111307
	s_ashr_i32 s15, s14, 31                                    // 000000001860: 860F9F0E
	s_lshl_b64 s[6:7], s[14:15], 2                             // 000000001864: 8486820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001868: BF870009
	s_add_u32 s6, s13, s6                                      // 00000000186C: 8006060D
	s_addc_u32 s7, s17, s7                                     // 000000001870: 82070711
	s_load_b32 s13, s[6:7], 0x10                               // 000000001874: F4000343 F8000010
	s_load_b32 s0, s[0:1], 0xc                                 // 00000000187C: F4000000 F800000C
	s_waitcnt lgkmcnt(0)                                       // 000000001884: BF89FC07
	v_fma_f32 v0, s11, s2, 0                                   // 000000001888: D6130000 0200040B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001890: BF870091
	v_fmac_f32_e64 v0, s10, s3                                 // 000000001894: D52B0000 0000060A
	v_fmac_f32_e64 v0, s12, s16                                // 00000000189C: D52B0000 0000200C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018A4: BF870001
	v_fmac_f32_e64 v0, s13, s0                                 // 0000000018A8: D52B0000 0000000D
	s_lshl_b64 s[0:1], s[8:9], 2                               // 0000000018B0: 84808208
	v_mov_b32_e32 v1, 0                                        // 0000000018B4: 7E020280
	s_add_u32 s2, s4, s0                                       // 0000000018B8: 80020004
	s_addc_u32 s3, s5, s1                                      // 0000000018BC: 82030105
	v_max_f32_e32 v0, 0, v0                                    // 0000000018C0: 20000080
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000018C4: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018C8: BF870009
	s_add_u32 s0, s2, s0                                       // 0000000018CC: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000018D0: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000018D4: DC6A0000 00000001
	s_nop 0                                                    // 0000000018DC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018E0: BFB60003
	s_endpgm                                                   // 0000000018E4: BFB00000
