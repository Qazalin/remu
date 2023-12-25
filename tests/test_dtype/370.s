
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n18>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_load_b32 s2, s[2:3], null                                // 000000001624: F4000081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	v_cvt_f32_i32_e32 v0, s2                                   // 000000001630: 7E000A02
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001634: BF870121
	v_cmp_ngt_f32_e64 s2, 0x48000000, |v0|                     // 000000001638: D41B0202 000200FF 48000000
	v_readfirstlane_b32 s3, v0                                 // 000000001644: 7E060500
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001648: 8B6A027E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000164C: BF870001
	s_and_b32 s2, s3, 0x7fffffff                               // 000000001650: 8B02FF03 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001658: BFA300A4 <E_3n18+0x2ec>
	s_and_b32 s3, s2, 0x7fffff                                 // 00000000165C: 8B03FF02 007FFFFF
	s_lshr_b32 s6, s2, 23                                      // 000000001664: 85069702
	s_bitset1_b32 s3, 23                                       // 000000001668: BE831297
	s_addk_i32 s6, 0xff88                                      // 00000000166C: B786FF88
	s_mul_hi_u32 s7, s3, 0xfe5163ab                            // 000000001670: 9687FF03 FE5163AB
	s_mul_i32 s8, s3, 0x3c439041                               // 000000001678: 9608FF03 3C439041
	s_mul_hi_u32 s9, s3, 0x3c439041                            // 000000001680: 9689FF03 3C439041
	s_add_u32 s7, s7, s8                                       // 000000001688: 80070807
	s_addc_u32 s8, 0, s9                                       // 00000000168C: 82080980
	s_mul_i32 s9, s3, 0xdb629599                               // 000000001690: 9609FF03 DB629599
	s_mul_hi_u32 s10, s3, 0xdb629599                           // 000000001698: 968AFF03 DB629599
	s_add_u32 s8, s8, s9                                       // 0000000016A0: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000016A4: 82090A80
	s_mul_i32 s10, s3, 0xf534ddc0                              // 0000000016A8: 960AFF03 F534DDC0
	s_mul_hi_u32 s11, s3, 0xf534ddc0                           // 0000000016B0: 968BFF03 F534DDC0
	s_add_u32 s9, s9, s10                                      // 0000000016B8: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000016BC: 820A0B80
	s_mul_i32 s11, s3, 0xfc2757d1                              // 0000000016C0: 960BFF03 FC2757D1
	s_mul_hi_u32 s12, s3, 0xfc2757d1                           // 0000000016C8: 968CFF03 FC2757D1
	s_add_u32 s10, s10, s11                                    // 0000000016D0: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000016D4: 820B0C80
	s_mul_i32 s12, s3, 0x4e441529                              // 0000000016D8: 960CFF03 4E441529
	s_mul_hi_u32 s13, s3, 0x4e441529                           // 0000000016E0: 968DFF03 4E441529
	s_add_u32 s11, s11, s12                                    // 0000000016E8: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000016EC: 820C0D80
	s_cmp_gt_u32 s6, 63                                        // 0000000016F0: BF08BF06
	s_mul_i32 s13, s3, 0xfe5163ab                              // 0000000016F4: 960DFF03 FE5163AB
	s_mul_hi_u32 s14, s3, 0xa2f9836e                           // 0000000016FC: 968EFF03 A2F9836E
	s_mul_i32 s3, s3, 0xa2f9836e                               // 000000001704: 9603FF03 A2F9836E
	s_cselect_b32 s15, s8, s10                                 // 00000000170C: 980F0A08
	s_cselect_b32 s7, s7, s9                                   // 000000001710: 98070907
	s_cselect_b32 s8, s13, s8                                  // 000000001714: 9808080D
	s_add_u32 s3, s12, s3                                      // 000000001718: 8003030C
	s_addc_u32 s12, 0, s14                                     // 00000000171C: 820C0E80
	s_cmp_gt_u32 s6, 63                                        // 000000001720: BF08BF06
	s_cselect_b32 s13, 0xffffffc0, 0                           // 000000001724: 980D80FF FFFFFFC0
	s_cselect_b32 s9, s9, s11                                  // 00000000172C: 98090B09
	s_cselect_b32 s3, s10, s3                                  // 000000001730: 9803030A
	s_cselect_b32 s10, s11, s12                                // 000000001734: 980A0C0B
	s_add_i32 s13, s13, s6                                     // 000000001738: 810D060D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000173C: BF870009
	s_cmp_gt_u32 s13, 31                                       // 000000001740: BF089F0D
	s_cselect_b32 s6, 0xffffffe0, 0                            // 000000001744: 980680FF FFFFFFE0
	s_cselect_b32 s11, s9, s3                                  // 00000000174C: 980B0309
	s_cselect_b32 s3, s3, s10                                  // 000000001750: 98030A03
	s_cselect_b32 s9, s15, s9                                  // 000000001754: 9809090F
	s_cselect_b32 s10, s7, s15                                 // 000000001758: 980A0F07
	s_cselect_b32 s7, s8, s7                                   // 00000000175C: 98070708
	s_add_i32 s6, s6, s13                                      // 000000001760: 81060D06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001764: BF870009
	s_cmp_gt_u32 s6, 31                                        // 000000001768: BF089F06
	s_cselect_b32 s8, 0xffffffe0, 0                            // 00000000176C: 980880FF FFFFFFE0
	s_cselect_b32 s3, s11, s3                                  // 000000001774: 9803030B
	s_cselect_b32 s11, s9, s11                                 // 000000001778: 980B0B09
	s_cselect_b32 s9, s10, s9                                  // 00000000177C: 9809090A
	s_cselect_b32 s7, s7, s10                                  // 000000001780: 98070A07
	s_add_i32 s8, s8, s6                                       // 000000001784: 81080608
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001788: BF8700C9
	s_sub_i32 s6, 32, s8                                       // 00000000178C: 818608A0
	s_cmp_eq_u32 s8, 0                                         // 000000001790: BF068008
	v_mov_b32_e32 v1, s6                                       // 000000001794: 7E020206
	s_cselect_b32 s8, -1, 0                                    // 000000001798: 980880C1
	v_alignbit_b32 v2, s3, s11, v1                             // 00000000179C: D6160002 04041603
	v_alignbit_b32 v3, s11, s9, v1                             // 0000000017A4: D6160003 0404120B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017AC: BF870112
	v_readfirstlane_b32 s6, v2                                 // 0000000017B0: 7E0C0502
	v_cndmask_b32_e64 v2, v3, s11, s8                          // 0000000017B4: D5010002 00201703
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017BC: BF870002
	s_cselect_b32 s3, s3, s6                                   // 0000000017C0: 98030603
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017C4: BF870481
	v_alignbit_b32 v3, s3, v2, 30                              // 0000000017C8: D6160003 027A0403
	s_bfe_u32 s10, s3, 0x1001d                                 // 0000000017D0: 930AFF03 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017D8: BF870009
	s_sub_i32 s6, 0, s10                                       // 0000000017DC: 81860A80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017E0: BF870481
	v_xor_b32_e32 v3, s6, v3                                   // 0000000017E4: 3A060606
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E8: BF870091
	v_clz_i32_u32_e32 v4, v3                                   // 0000000017EC: 7E087303
	v_min_u32_e32 v4, 32, v4                                   // 0000000017F0: 260808A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000017F4: BF870131
	v_lshlrev_b32_e32 v6, 23, v4                               // 0000000017F8: 300C0897
	v_alignbit_b32 v1, s9, s7, v1                              // 0000000017FC: D6160001 04040E09
	v_sub_nc_u32_e32 v5, 31, v4                                // 000000001804: 4C0A089F
	v_cndmask_b32_e64 v1, v1, s9, s8                           // 000000001808: D5010001 00201301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001810: BF870121
	v_alignbit_b32 v2, v2, v1, 30                              // 000000001814: D6160002 027A0302
	v_alignbit_b32 v1, v1, s7, 30                              // 00000000181C: D6160001 02780F01
	v_xor_b32_e32 v2, s6, v2                                   // 000000001824: 3A040406
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001828: BF870002
	v_xor_b32_e32 v1, s6, v1                                   // 00000000182C: 3A020206
	s_lshr_b32 s6, s3, 29                                      // 000000001830: 85069D03
	s_lshr_b32 s3, s3, 30                                      // 000000001834: 85039E03
	s_lshl_b32 s6, s6, 31                                      // 000000001838: 84069F06
	v_alignbit_b32 v3, v3, v2, v5                              // 00000000183C: D6160003 04160503
	v_alignbit_b32 v1, v2, v1, v5                              // 000000001844: D6160001 04160302
	s_or_b32 s7, s6, 0.5                                       // 00000000184C: 8C07F006
	s_add_i32 s3, s10, s3                                      // 000000001850: 8103030A
	v_sub_nc_u32_e32 v6, s7, v6                                // 000000001854: 4C0C0C07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001858: BF870092
	v_alignbit_b32 v2, v3, v1, 9                               // 00000000185C: D6160002 02260303
	v_clz_i32_u32_e32 v5, v2                                   // 000000001864: 7E0A7302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001868: BF870091
	v_min_u32_e32 v5, 32, v5                                   // 00000000186C: 260A0AA0
	v_sub_nc_u32_e32 v7, 31, v5                                // 000000001870: 4C0E0A9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001874: BF870121
	v_alignbit_b32 v1, v2, v1, v7                              // 000000001878: D6160001 041E0302
	v_lshrrev_b32_e32 v2, 9, v3                                // 000000001880: 32040689
	v_lshrrev_b32_e32 v1, 9, v1                                // 000000001884: 32020289
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001888: BF8700A2
	v_or_b32_e32 v2, v2, v6                                    // 00000000188C: 38040D02
	v_add_nc_u32_e32 v4, v5, v4                                // 000000001890: 4A080905
	v_lshlrev_b32_e32 v3, 23, v4                               // 000000001894: 30060897
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001898: BF870211
	v_sub_nc_u32_e32 v1, v1, v3                                // 00000000189C: 4C020701
	v_mul_f32_e32 v3, 0x3fc90fda, v2                           // 0000000018A0: 100604FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018A8: BF870112
	v_add_nc_u32_e32 v1, 0x33000000, v1                        // 0000000018AC: 4A0202FF 33000000
	v_fma_f32 v4, 0x3fc90fda, v2, -v3                          // 0000000018B4: D6130004 840E04FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018C0: BF870112
	v_or_b32_e32 v1, s6, v1                                    // 0000000018C4: 38020206
	v_fmac_f32_e32 v4, 0x33a22168, v2                          // 0000000018C8: 560804FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D0: BF870091
	v_fmac_f32_e32 v4, 0x3fc90fda, v1                          // 0000000018D4: 560802FF 3FC90FDA
	v_add_f32_e32 v1, v3, v4                                   // 0000000018DC: 06020903
	s_cbranch_execz 2                                          // 0000000018E0: BFA50002 <E_3n18+0x2ec>
	v_mov_b32_e32 v2, s3                                       // 0000000018E4: 7E040203
	s_branch 14                                                // 0000000018E8: BFA0000E <E_3n18+0x324>
	v_mul_f32_e64 v1, 0x3f22f983, s2                           // 0000000018EC: D5080001 000004FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F8: BF870091
	v_rndne_f32_e32 v2, v1                                     // 0000000018FC: 7E044701
	v_fma_f32 v1, 0xbfc90fda, v2, s2                           // 000000001900: D6130001 000A04FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000190C: BF870091
	v_fmac_f32_e32 v1, 0xb3a22168, v2                          // 000000001910: 560204FF B3A22168
	v_fmac_f32_e32 v1, 0xa7c234c4, v2                          // 000000001918: 560204FF A7C234C4
	v_cvt_i32_f32_e32 v2, v2                                   // 000000001920: 7E041102
	v_sub_f32_e32 v3, 0x3fc90fdb, v0                           // 000000001924: 080600FF 3FC90FDB
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000192C: BF870121
	v_cmp_ngt_f32_e64 s3, 0x48000000, |v3|                     // 000000001930: D41B0203 000206FF 48000000
	v_readfirstlane_b32 s6, v3                                 // 00000000193C: 7E0C0503
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001940: 8B6A037E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001944: BF870001
	s_and_b32 s3, s6, 0x7fffffff                               // 000000001948: 8B03FF06 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001950: BFA300A4 <E_3n18+0x5e4>
	s_and_b32 s6, s3, 0x7fffff                                 // 000000001954: 8B06FF03 007FFFFF
	s_lshr_b32 s7, s3, 23                                      // 00000000195C: 85079703
	s_bitset1_b32 s6, 23                                       // 000000001960: BE861297
	s_addk_i32 s7, 0xff88                                      // 000000001964: B787FF88
	s_mul_hi_u32 s8, s6, 0xfe5163ab                            // 000000001968: 9688FF06 FE5163AB
	s_mul_i32 s9, s6, 0x3c439041                               // 000000001970: 9609FF06 3C439041
	s_mul_hi_u32 s10, s6, 0x3c439041                           // 000000001978: 968AFF06 3C439041
	s_add_u32 s8, s8, s9                                       // 000000001980: 80080908
	s_addc_u32 s9, 0, s10                                      // 000000001984: 82090A80
	s_mul_i32 s10, s6, 0xdb629599                              // 000000001988: 960AFF06 DB629599
	s_mul_hi_u32 s11, s6, 0xdb629599                           // 000000001990: 968BFF06 DB629599
	s_add_u32 s9, s9, s10                                      // 000000001998: 80090A09
	s_addc_u32 s10, 0, s11                                     // 00000000199C: 820A0B80
	s_mul_i32 s11, s6, 0xf534ddc0                              // 0000000019A0: 960BFF06 F534DDC0
	s_mul_hi_u32 s12, s6, 0xf534ddc0                           // 0000000019A8: 968CFF06 F534DDC0
	s_add_u32 s10, s10, s11                                    // 0000000019B0: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000019B4: 820B0C80
	s_mul_i32 s12, s6, 0xfc2757d1                              // 0000000019B8: 960CFF06 FC2757D1
	s_mul_hi_u32 s13, s6, 0xfc2757d1                           // 0000000019C0: 968DFF06 FC2757D1
	s_add_u32 s11, s11, s12                                    // 0000000019C8: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000019CC: 820C0D80
	s_mul_i32 s13, s6, 0x4e441529                              // 0000000019D0: 960DFF06 4E441529
	s_mul_hi_u32 s14, s6, 0x4e441529                           // 0000000019D8: 968EFF06 4E441529
	s_add_u32 s12, s12, s13                                    // 0000000019E0: 800C0D0C
	s_addc_u32 s13, 0, s14                                     // 0000000019E4: 820D0E80
	s_cmp_gt_u32 s7, 63                                        // 0000000019E8: BF08BF07
	s_mul_i32 s14, s6, 0xfe5163ab                              // 0000000019EC: 960EFF06 FE5163AB
	s_mul_hi_u32 s15, s6, 0xa2f9836e                           // 0000000019F4: 968FFF06 A2F9836E
	s_mul_i32 s6, s6, 0xa2f9836e                               // 0000000019FC: 9606FF06 A2F9836E
	s_cselect_b32 s16, s9, s11                                 // 000000001A04: 98100B09
	s_cselect_b32 s8, s8, s10                                  // 000000001A08: 98080A08
	s_cselect_b32 s9, s14, s9                                  // 000000001A0C: 9809090E
	s_add_u32 s6, s13, s6                                      // 000000001A10: 8006060D
	s_addc_u32 s13, 0, s15                                     // 000000001A14: 820D0F80
	s_cmp_gt_u32 s7, 63                                        // 000000001A18: BF08BF07
	s_cselect_b32 s14, 0xffffffc0, 0                           // 000000001A1C: 980E80FF FFFFFFC0
	s_cselect_b32 s10, s10, s12                                // 000000001A24: 980A0C0A
	s_cselect_b32 s6, s11, s6                                  // 000000001A28: 9806060B
	s_cselect_b32 s11, s12, s13                                // 000000001A2C: 980B0D0C
	s_add_i32 s14, s14, s7                                     // 000000001A30: 810E070E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A34: BF870009
	s_cmp_gt_u32 s14, 31                                       // 000000001A38: BF089F0E
	s_cselect_b32 s7, 0xffffffe0, 0                            // 000000001A3C: 980780FF FFFFFFE0
	s_cselect_b32 s12, s10, s6                                 // 000000001A44: 980C060A
	s_cselect_b32 s6, s6, s11                                  // 000000001A48: 98060B06
	s_cselect_b32 s10, s16, s10                                // 000000001A4C: 980A0A10
	s_cselect_b32 s11, s8, s16                                 // 000000001A50: 980B1008
	s_cselect_b32 s8, s9, s8                                   // 000000001A54: 98080809
	s_add_i32 s7, s7, s14                                      // 000000001A58: 81070E07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A5C: BF870009
	s_cmp_gt_u32 s7, 31                                        // 000000001A60: BF089F07
	s_cselect_b32 s9, 0xffffffe0, 0                            // 000000001A64: 980980FF FFFFFFE0
	s_cselect_b32 s6, s12, s6                                  // 000000001A6C: 9806060C
	s_cselect_b32 s12, s10, s12                                // 000000001A70: 980C0C0A
	s_cselect_b32 s10, s11, s10                                // 000000001A74: 980A0A0B
	s_cselect_b32 s8, s8, s11                                  // 000000001A78: 98080B08
	s_add_i32 s9, s9, s7                                       // 000000001A7C: 81090709
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001A80: BF8700C9
	s_sub_i32 s7, 32, s9                                       // 000000001A84: 818709A0
	s_cmp_eq_u32 s9, 0                                         // 000000001A88: BF068009
	v_mov_b32_e32 v4, s7                                       // 000000001A8C: 7E080207
	s_cselect_b32 s9, -1, 0                                    // 000000001A90: 980980C1
	v_alignbit_b32 v5, s6, s12, v4                             // 000000001A94: D6160005 04101806
	v_alignbit_b32 v6, s12, s10, v4                            // 000000001A9C: D6160006 0410140C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001AA4: BF870112
	v_readfirstlane_b32 s7, v5                                 // 000000001AA8: 7E0E0505
	v_cndmask_b32_e64 v5, v6, s12, s9                          // 000000001AAC: D5010005 00241906
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001AB4: BF870002
	s_cselect_b32 s6, s6, s7                                   // 000000001AB8: 98060706
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001ABC: BF870481
	v_alignbit_b32 v6, s6, v5, 30                              // 000000001AC0: D6160006 027A0A06
	s_bfe_u32 s11, s6, 0x1001d                                 // 000000001AC8: 930BFF06 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001AD0: BF870009
	s_sub_i32 s7, 0, s11                                       // 000000001AD4: 81870B80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001AD8: BF870481
	v_xor_b32_e32 v6, s7, v6                                   // 000000001ADC: 3A0C0C07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AE0: BF870091
	v_clz_i32_u32_e32 v7, v6                                   // 000000001AE4: 7E0E7306
	v_min_u32_e32 v7, 32, v7                                   // 000000001AE8: 260E0EA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001AEC: BF870131
	v_lshlrev_b32_e32 v9, 23, v7                               // 000000001AF0: 30120E97
	v_alignbit_b32 v4, s10, s8, v4                             // 000000001AF4: D6160004 0410100A
	v_sub_nc_u32_e32 v8, 31, v7                                // 000000001AFC: 4C100E9F
	v_cndmask_b32_e64 v4, v4, s10, s9                          // 000000001B00: D5010004 00241504
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001B08: BF870121
	v_alignbit_b32 v5, v5, v4, 30                              // 000000001B0C: D6160005 027A0905
	v_alignbit_b32 v4, v4, s8, 30                              // 000000001B14: D6160004 02781104
	v_xor_b32_e32 v5, s7, v5                                   // 000000001B1C: 3A0A0A07
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B20: BF870002
	v_xor_b32_e32 v4, s7, v4                                   // 000000001B24: 3A080807
	s_lshr_b32 s7, s6, 29                                      // 000000001B28: 85079D06
	s_lshr_b32 s6, s6, 30                                      // 000000001B2C: 85069E06
	s_lshl_b32 s7, s7, 31                                      // 000000001B30: 84079F07
	v_alignbit_b32 v6, v6, v5, v8                              // 000000001B34: D6160006 04220B06
	v_alignbit_b32 v4, v5, v4, v8                              // 000000001B3C: D6160004 04220905
	s_or_b32 s8, s7, 0.5                                       // 000000001B44: 8C08F007
	s_add_i32 s6, s11, s6                                      // 000000001B48: 8106060B
	v_sub_nc_u32_e32 v9, s8, v9                                // 000000001B4C: 4C121208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B50: BF870092
	v_alignbit_b32 v5, v6, v4, 9                               // 000000001B54: D6160005 02260906
	v_clz_i32_u32_e32 v8, v5                                   // 000000001B5C: 7E107305
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B60: BF870091
	v_min_u32_e32 v8, 32, v8                                   // 000000001B64: 261010A0
	v_sub_nc_u32_e32 v10, 31, v8                               // 000000001B68: 4C14109F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001B6C: BF870121
	v_alignbit_b32 v4, v5, v4, v10                             // 000000001B70: D6160004 042A0905
	v_lshrrev_b32_e32 v5, 9, v6                                // 000000001B78: 320A0C89
	v_lshrrev_b32_e32 v4, 9, v4                                // 000000001B7C: 32080889
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B80: BF8700A2
	v_or_b32_e32 v5, v5, v9                                    // 000000001B84: 380A1305
	v_add_nc_u32_e32 v7, v8, v7                                // 000000001B88: 4A0E0F08
	v_lshlrev_b32_e32 v6, 23, v7                               // 000000001B8C: 300C0E97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001B90: BF870211
	v_sub_nc_u32_e32 v4, v4, v6                                // 000000001B94: 4C080D04
	v_mul_f32_e32 v6, 0x3fc90fda, v5                           // 000000001B98: 100C0AFF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001BA0: BF870112
	v_add_nc_u32_e32 v4, 0x33000000, v4                        // 000000001BA4: 4A0808FF 33000000
	v_fma_f32 v7, 0x3fc90fda, v5, -v6                          // 000000001BAC: D6130007 841A0AFF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001BB8: BF870112
	v_or_b32_e32 v4, s7, v4                                    // 000000001BBC: 38080807
	v_fmac_f32_e32 v7, 0x33a22168, v5                          // 000000001BC0: 560E0AFF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC8: BF870091
	v_fmac_f32_e32 v7, 0x3fc90fda, v4                          // 000000001BCC: 560E08FF 3FC90FDA
	v_add_f32_e32 v4, v6, v7                                   // 000000001BD4: 06080F06
	s_cbranch_execz 2                                          // 000000001BD8: BFA50002 <E_3n18+0x5e4>
	v_mov_b32_e32 v5, s6                                       // 000000001BDC: 7E0A0206
	s_branch 14                                                // 000000001BE0: BFA0000E <E_3n18+0x61c>
	v_mul_f32_e64 v4, 0x3f22f983, s3                           // 000000001BE4: D5080004 000006FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BF0: BF870091
	v_rndne_f32_e32 v5, v4                                     // 000000001BF4: 7E0A4704
	v_fma_f32 v4, 0xbfc90fda, v5, s3                           // 000000001BF8: D6130004 000E0AFF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C04: BF870091
	v_fmac_f32_e32 v4, 0xb3a22168, v5                          // 000000001C08: 56080AFF B3A22168
	v_fmac_f32_e32 v4, 0xa7c234c4, v5                          // 000000001C10: 56080AFF A7C234C4
	v_cvt_i32_f32_e32 v5, v5                                   // 000000001C18: 7E0A1105
	v_dual_mul_f32 v6, v1, v1 :: v_dual_lshlrev_b32 v7, 30, v2 // 000000001C1C: C8E20301 0606049E
	s_mov_b32 s6, 0xb94c1982                                   // 000000001C24: BE8600FF B94C1982
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001C2C: BF870123
	v_mul_f32_e32 v8, v4, v4                                   // 000000001C30: 10100904
	s_mov_b32 s7, 0x37d75334                                   // 000000001C34: BE8700FF 37D75334
	v_fmaak_f32 v9, s6, v6, 0x3c0881c4                         // 000000001C3C: 5A120C06 3C0881C4
	v_xor_b32_e32 v0, s2, v0                                   // 000000001C44: 3A000002
	v_and_b32_e32 v7, 0x80000000, v7                           // 000000001C48: 360E0EFF 80000000
	v_dual_fmaak_f32 v11, s6, v8, 0x3c0881c4 :: v_dual_and_b32 v2, 1, v2// 000000001C50: C8641006 0B020481 3C0881C4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001C5C: BF870224
	v_fmaak_f32 v9, v6, v9, 0xbe2aaa9d                         // 000000001C60: 5A121306 BE2AAA9D
	v_fmaak_f32 v10, s7, v6, 0xbab64f3b                        // 000000001C68: 5A140C07 BAB64F3B
	v_xor_b32_e32 v0, v0, v7                                   // 000000001C70: 3A000F00
	v_xor_b32_e32 v3, s3, v3                                   // 000000001C74: 3A060603
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 000000001C78: 7C940480
	v_mul_f32_e32 v9, v6, v9                                   // 000000001C7C: 10121306
	v_fmaak_f32 v7, v6, v10, 0x3d2aabf7                        // 000000001C80: 5A0E1506 3D2AABF7
	v_fmaak_f32 v10, v8, v11, 0xbe2aaa9d                       // 000000001C88: 5A141708 BE2AAA9D
	v_fmaak_f32 v12, s7, v8, 0xbab64f3b                        // 000000001C90: 5A181007 BAB64F3B
	s_add_u32 s0, s0, s4                                       // 000000001C98: 80000400
	v_fmac_f32_e32 v1, v1, v9                                  // 000000001C9C: 56021301
	s_addc_u32 s1, s1, s5                                      // 000000001CA0: 82010501
	v_dual_mul_f32 v10, v8, v10 :: v_dual_fmaak_f32 v7, v6, v7, 0xbf000004// 000000001CA4: C8C21508 0A060F06 BF000004
	v_fmaak_f32 v11, v8, v12, 0x3d2aabf7                       // 000000001CB0: 5A161908 3D2AABF7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001CB8: BF870192
	v_fmac_f32_e32 v4, v4, v10                                 // 000000001CBC: 56081504
	v_fma_f32 v6, v6, v7, 1.0                                  // 000000001CC0: D6130006 03CA0F06
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CC8: BF870091
	v_cndmask_b32_e32 v1, v6, v1, vcc_lo                       // 000000001CCC: 02020306
	v_xor_b32_e32 v0, v0, v1                                   // 000000001CD0: 3A000300
	v_lshlrev_b32_e32 v9, 30, v5                               // 000000001CD4: 30120A9E
	v_and_b32_e32 v5, 1, v5                                    // 000000001CD8: 360A0A81
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001CDC: BF8700A1
	v_cmp_eq_u32_e32 vcc_lo, 0, v5                             // 000000001CE0: 7C940A80
	v_fmaak_f32 v11, v8, v11, 0xbf000004                       // 000000001CE4: 5A161708 BF000004
	v_fma_f32 v7, v8, v11, 1.0                                 // 000000001CEC: D6130007 03CA1708
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)// 000000001CF4: BF870151
	v_cndmask_b32_e32 v2, v7, v4, vcc_lo                       // 000000001CF8: 02040907
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 000000001CFC: D47E006A 0001FE02 000001F8
	v_cndmask_b32_e32 v0, 0x7fc00000, v0, vcc_lo               // 000000001D08: 020000FF 7FC00000
	v_and_b32_e32 v8, 0x80000000, v9                           // 000000001D10: 361012FF 80000000
	v_cmp_class_f32_e64 vcc_lo, s3, 0x1f8                      // 000000001D18: D47E006A 0001FE03 000001F8
	v_xor_b32_e32 v3, v3, v8                                   // 000000001D24: 3A061103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D28: BF870091
	v_xor_b32_e32 v1, v3, v2                                   // 000000001D2C: 3A020503
	v_cndmask_b32_e32 v1, 0x7fc00000, v1, vcc_lo               // 000000001D30: 020202FF 7FC00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001D38: BF870121
	v_div_scale_f32 v2, null, v1, v1, v0                       // 000000001D3C: D6FC7C02 04020301
	v_div_scale_f32 v5, vcc_lo, v0, v1, v0                     // 000000001D44: D6FC6A05 04020300
	v_rcp_f32_e32 v3, v2                                       // 000000001D4C: 7E065502
	s_waitcnt_depctr 0xfff                                     // 000000001D50: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 000000001D54: D6130004 23CA0702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D5C: BF870091
	v_fmac_f32_e32 v3, v4, v3                                  // 000000001D60: 56060704
	v_mul_f32_e32 v4, v5, v3                                   // 000000001D64: 10080705
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D68: BF870091
	v_fma_f32 v6, -v2, v4, v5                                  // 000000001D6C: D6130006 24160902
	v_fmac_f32_e32 v4, v6, v3                                  // 000000001D74: 56080706
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D78: BF870091
	v_fma_f32 v2, -v2, v4, v5                                  // 000000001D7C: D6130002 24160902
	v_div_fmas_f32 v2, v2, v3, v4                              // 000000001D84: D6370002 04120702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001D8C: BF870001
	v_div_fixup_f32 v0, v2, v1, v0                             // 000000001D90: D6270000 04020302
	v_mov_b32_e32 v1, 0                                        // 000000001D98: 7E020280
	global_store_b32 v1, v0, s[0:1]                            // 000000001D9C: DC6A0000 00000001
	s_nop 0                                                    // 000000001DA4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001DA8: BFB60003
	s_endpgm                                                   // 000000001DAC: BFB00000
