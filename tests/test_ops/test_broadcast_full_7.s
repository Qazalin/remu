
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_2_3_5_7_8n3>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_ashr_i32 s2, s13, 31                                     // 000000001708: 86029F0D
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshr_b32 s2, s2, 29                                      // 000000001714: 85029D02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001718: BF870499
	s_add_i32 s10, s13, s2                                     // 00000000171C: 810A020D
	s_ashr_i32 s8, s10, 3                                      // 000000001720: 8608830A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001724: BF870499
	s_mul_hi_i32 s2, s8, 0x92492493                            // 000000001728: 9702FF08 92492493
	s_add_i32 s2, s2, s8                                       // 000000001730: 81020802
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001734: BF870009
	s_lshr_b32 s3, s2, 31                                      // 000000001738: 85039F02
	s_ashr_i32 s9, s2, 2                                       // 00000000173C: 86098202
	s_mul_i32 s2, s14, 7                                       // 000000001740: 9602870E
	s_add_i32 s9, s9, s3                                       // 000000001744: 81090309
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_mul_i32 s9, s9, 7                                        // 00000000174C: 96098709
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001750: 84828202
	s_sub_i32 s8, s8, s9                                       // 000000001754: 81880908
	s_waitcnt lgkmcnt(0)                                       // 000000001758: BF89FC07
	s_add_u32 s6, s6, s2                                       // 00000000175C: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001760: 82070307
	s_ashr_i32 s9, s8, 31                                      // 000000001764: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001768: BF870009
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000176C: 84828208
	s_mul_hi_i32 s9, s13, 0x92492493                           // 000000001770: 9709FF0D 92492493
	s_add_u32 s2, s6, s2                                       // 000000001778: 80020206
	s_mul_i32 s6, s15, 40                                      // 00000000177C: 9606A80F
	s_addc_u32 s3, s7, s3                                      // 000000001780: 82030307
	s_add_i32 s9, s9, s13                                      // 000000001784: 81090D09
	s_ashr_i32 s7, s6, 31                                      // 000000001788: 86079F06
	s_lshr_b32 s11, s9, 31                                     // 00000000178C: 850B9F09
	s_ashr_i32 s9, s9, 5                                       // 000000001790: 86098509
	s_and_b32 s10, s10, -8                                     // 000000001794: 8B0AC80A
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001798: 84868206
	s_add_i32 s9, s9, s11                                      // 00000000179C: 81090B09
	s_sub_i32 s10, s13, s10                                    // 0000000017A0: 818A0A0D
	s_add_u32 s6, s0, s6                                       // 0000000017A4: 80060600
	s_addc_u32 s7, s1, s7                                      // 0000000017A8: 82070701
	s_lshl_b32 s0, s9, 3                                       // 0000000017AC: 84008309
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017B0: BF870499
	s_ashr_i32 s1, s0, 31                                      // 0000000017B4: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017B8: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017BC: BF8704B9
	s_add_u32 s6, s6, s0                                       // 0000000017C0: 80060006
	s_addc_u32 s7, s7, s1                                      // 0000000017C4: 82070107
	s_ashr_i32 s11, s10, 31                                    // 0000000017C8: 860B9F0A
	s_lshl_b64 s[0:1], s[10:11], 2                             // 0000000017CC: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017D0: BF870009
	s_add_u32 s6, s6, s0                                       // 0000000017D4: 80060006
	s_addc_u32 s7, s7, s1                                      // 0000000017D8: 82070107
	s_load_b32 s10, s[2:3], null                               // 0000000017DC: F4000281 F8000000
	s_load_b32 s11, s[6:7], null                               // 0000000017E4: F40002C3 F8000000
	s_mul_i32 s2, s15, 0x348                                   // 0000000017EC: 9602FF0F 00000348
	s_mul_i32 s6, s14, 0x118                                   // 0000000017F4: 9606FF0E 00000118
	s_ashr_i32 s3, s2, 31                                      // 0000000017FC: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001800: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001804: 84828202
	s_add_u32 s12, s4, s2                                      // 000000001808: 800C0204
	s_addc_u32 s5, s5, s3                                      // 00000000180C: 82050305
	s_ashr_i32 s7, s6, 31                                      // 000000001810: 86079F06
	s_mul_i32 s4, s9, 56                                       // 000000001814: 9604B809
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001818: 84828206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000181C: BF8704B9
	s_add_u32 s6, s12, s2                                      // 000000001820: 8006020C
	s_addc_u32 s7, s5, s3                                      // 000000001824: 82070305
	s_ashr_i32 s5, s4, 31                                      // 000000001828: 86059F04
	s_lshl_b64 s[2:3], s[4:5], 2                               // 00000000182C: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)// 000000001830: BF870159
	s_add_u32 s4, s6, s2                                       // 000000001834: 80040206
	s_waitcnt lgkmcnt(0)                                       // 000000001838: BF89FC07
	v_div_scale_f32 v0, null, s11, s11, s10                    // 00000000183C: D6FC7C00 0028160B
	v_div_scale_f32 v3, vcc_lo, s10, s11, s10                  // 000000001844: D6FC6A03 0028160A
	s_addc_u32 s5, s7, s3                                      // 00000000184C: 82050307
	v_rcp_f32_e32 v1, v0                                       // 000000001850: 7E025500
	s_lshl_b32 s2, s8, 3                                       // 000000001854: 84028308
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001858: BF870499
	s_ashr_i32 s3, s2, 31                                      // 00000000185C: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001860: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001864: BF870009
	s_add_u32 s2, s4, s2                                       // 000000001868: 80020204
	s_addc_u32 s3, s5, s3                                      // 00000000186C: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001870: 80000002
	s_waitcnt_depctr 0xfff                                     // 000000001874: BF880FFF
	v_fma_f32 v2, -v0, v1, 1.0                                 // 000000001878: D6130002 23CA0300
	s_addc_u32 s1, s3, s1                                      // 000000001880: 82010103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001884: BF870091
	v_fmac_f32_e32 v1, v2, v1                                  // 000000001888: 56020302
	v_mul_f32_e32 v2, v3, v1                                   // 00000000188C: 10040303
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001890: BF870091
	v_fma_f32 v4, -v0, v2, v3                                  // 000000001894: D6130004 240E0500
	v_fmac_f32_e32 v2, v4, v1                                  // 00000000189C: 56040304
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018A0: BF870091
	v_fma_f32 v0, -v0, v2, v3                                  // 0000000018A4: D6130000 240E0500
	v_div_fmas_f32 v0, v0, v1, v2                              // 0000000018AC: D6370000 040A0300
	v_mov_b32_e32 v1, 0                                        // 0000000018B4: 7E020280
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000018B8: BF870002
	v_div_fixup_f32 v0, v0, s11, s10                           // 0000000018BC: D6270000 00281700
	global_store_b32 v1, v0, s[0:1]                            // 0000000018C4: DC6A0000 00000001
	s_nop 0                                                    // 0000000018CC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018D0: BFB60003
	s_endpgm                                                   // 0000000018D4: BFB00000
