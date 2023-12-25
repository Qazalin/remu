
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n67>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[6:7], s[4:5], 3                               // 000000001614: 84868304
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b64 s[2:3], s[2:3], null                            // 000000001624: F4040081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	s_xor_b32 s6, s2, s3                                       // 000000001630: 8D060302
	s_cls_i32 s7, s3                                           // 000000001634: BE870C03
	s_ashr_i32 s6, s6, 31                                      // 000000001638: 86069F06
	s_add_i32 s7, s7, -1                                       // 00000000163C: 8107C107
	s_add_i32 s6, s6, 32                                       // 000000001640: 8106A006
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001644: BF870499
	s_min_u32 s6, s7, s6                                       // 000000001648: 89860607
	s_lshl_b64 s[2:3], s[2:3], s6                              // 00000000164C: 84820602
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001650: BF870499
	s_min_u32 s2, s2, 1                                        // 000000001654: 89828102
	s_or_b32 s2, s3, s2                                        // 000000001658: 8C020203
	s_mov_b32 s3, -1                                           // 00000000165C: BE8300C1
	v_cvt_f32_i32_e32 v0, s2                                   // 000000001660: 7E000A02
	s_sub_i32 s2, 32, s6                                       // 000000001664: 818206A0
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001668: BF870481
	v_ldexp_f32 v0, v0, s2                                     // 00000000166C: D71C0000 00000500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001674: BF870091
	v_sub_f32_e32 v0, 0x3fc90fdb, v0                           // 000000001678: 080000FF 3FC90FDB
	v_cmp_ngt_f32_e64 s2, 0x48000000, |v0|                     // 000000001680: D41B0202 000200FF 48000000
	v_readfirstlane_b32 s6, v0                                 // 00000000168C: 7E0C0500
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001690: BF870092
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001694: 8B6A027E
	s_and_b32 s2, s6, 0x7fffffff                               // 000000001698: 8B02FF06 7FFFFFFF
	s_cbranch_vccz 166                                         // 0000000016A0: BFA300A6 <E_3n67+0x33c>
	s_and_b32 s3, s2, 0x7fffff                                 // 0000000016A4: 8B03FF02 007FFFFF
	s_lshr_b32 s6, s2, 23                                      // 0000000016AC: 85069702
	s_bitset1_b32 s3, 23                                       // 0000000016B0: BE831297
	s_addk_i32 s6, 0xff88                                      // 0000000016B4: B786FF88
	s_mul_hi_u32 s7, s3, 0xfe5163ab                            // 0000000016B8: 9687FF03 FE5163AB
	s_mul_i32 s8, s3, 0x3c439041                               // 0000000016C0: 9608FF03 3C439041
	s_mul_hi_u32 s9, s3, 0x3c439041                            // 0000000016C8: 9689FF03 3C439041
	s_add_u32 s7, s7, s8                                       // 0000000016D0: 80070807
	s_addc_u32 s8, 0, s9                                       // 0000000016D4: 82080980
	s_mul_i32 s9, s3, 0xdb629599                               // 0000000016D8: 9609FF03 DB629599
	s_mul_hi_u32 s10, s3, 0xdb629599                           // 0000000016E0: 968AFF03 DB629599
	s_add_u32 s8, s8, s9                                       // 0000000016E8: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000016EC: 82090A80
	s_mul_i32 s10, s3, 0xf534ddc0                              // 0000000016F0: 960AFF03 F534DDC0
	s_mul_hi_u32 s11, s3, 0xf534ddc0                           // 0000000016F8: 968BFF03 F534DDC0
	s_add_u32 s9, s9, s10                                      // 000000001700: 80090A09
	s_addc_u32 s10, 0, s11                                     // 000000001704: 820A0B80
	s_mul_i32 s11, s3, 0xfc2757d1                              // 000000001708: 960BFF03 FC2757D1
	s_mul_hi_u32 s12, s3, 0xfc2757d1                           // 000000001710: 968CFF03 FC2757D1
	s_add_u32 s10, s10, s11                                    // 000000001718: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 00000000171C: 820B0C80
	s_mul_i32 s12, s3, 0x4e441529                              // 000000001720: 960CFF03 4E441529
	s_mul_hi_u32 s13, s3, 0x4e441529                           // 000000001728: 968DFF03 4E441529
	s_add_u32 s11, s11, s12                                    // 000000001730: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 000000001734: 820C0D80
	s_cmp_gt_u32 s6, 63                                        // 000000001738: BF08BF06
	s_mul_i32 s13, s3, 0xfe5163ab                              // 00000000173C: 960DFF03 FE5163AB
	s_mul_hi_u32 s14, s3, 0xa2f9836e                           // 000000001744: 968EFF03 A2F9836E
	s_mul_i32 s3, s3, 0xa2f9836e                               // 00000000174C: 9603FF03 A2F9836E
	s_cselect_b32 s15, s8, s10                                 // 000000001754: 980F0A08
	s_cselect_b32 s7, s7, s9                                   // 000000001758: 98070907
	s_cselect_b32 s8, s13, s8                                  // 00000000175C: 9808080D
	s_add_u32 s3, s12, s3                                      // 000000001760: 8003030C
	s_addc_u32 s12, 0, s14                                     // 000000001764: 820C0E80
	s_cmp_gt_u32 s6, 63                                        // 000000001768: BF08BF06
	s_cselect_b32 s13, 0xffffffc0, 0                           // 00000000176C: 980D80FF FFFFFFC0
	s_cselect_b32 s9, s9, s11                                  // 000000001774: 98090B09
	s_cselect_b32 s3, s10, s3                                  // 000000001778: 9803030A
	s_cselect_b32 s10, s11, s12                                // 00000000177C: 980A0C0B
	s_add_i32 s13, s13, s6                                     // 000000001780: 810D060D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_cmp_gt_u32 s13, 31                                       // 000000001788: BF089F0D
	s_cselect_b32 s6, 0xffffffe0, 0                            // 00000000178C: 980680FF FFFFFFE0
	s_cselect_b32 s11, s9, s3                                  // 000000001794: 980B0309
	s_cselect_b32 s3, s3, s10                                  // 000000001798: 98030A03
	s_cselect_b32 s9, s15, s9                                  // 00000000179C: 9809090F
	s_cselect_b32 s10, s7, s15                                 // 0000000017A0: 980A0F07
	s_cselect_b32 s7, s8, s7                                   // 0000000017A4: 98070708
	s_add_i32 s6, s6, s13                                      // 0000000017A8: 81060D06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017AC: BF870009
	s_cmp_gt_u32 s6, 31                                        // 0000000017B0: BF089F06
	s_cselect_b32 s8, 0xffffffe0, 0                            // 0000000017B4: 980880FF FFFFFFE0
	s_cselect_b32 s3, s11, s3                                  // 0000000017BC: 9803030B
	s_cselect_b32 s11, s9, s11                                 // 0000000017C0: 980B0B09
	s_cselect_b32 s9, s10, s9                                  // 0000000017C4: 9809090A
	s_cselect_b32 s7, s7, s10                                  // 0000000017C8: 98070A07
	s_add_i32 s8, s8, s6                                       // 0000000017CC: 81080608
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017D0: BF8700C9
	s_sub_i32 s6, 32, s8                                       // 0000000017D4: 818608A0
	s_cmp_eq_u32 s8, 0                                         // 0000000017D8: BF068008
	v_mov_b32_e32 v1, s6                                       // 0000000017DC: 7E020206
	s_cselect_b32 s8, -1, 0                                    // 0000000017E0: 980880C1
	v_alignbit_b32 v2, s3, s11, v1                             // 0000000017E4: D6160002 04041603
	v_alignbit_b32 v3, s11, s9, v1                             // 0000000017EC: D6160003 0404120B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017F4: BF870112
	v_readfirstlane_b32 s6, v2                                 // 0000000017F8: 7E0C0502
	v_cndmask_b32_e64 v2, v3, s11, s8                          // 0000000017FC: D5010002 00201703
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001804: BF870002
	s_cselect_b32 s3, s3, s6                                   // 000000001808: 98030603
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 00000000180C: BF870481
	v_alignbit_b32 v3, s3, v2, 30                              // 000000001810: D6160003 027A0403
	s_bfe_u32 s6, s3, 0x1001d                                  // 000000001818: 9306FF03 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001820: BF870009
	s_sub_i32 s10, 0, s6                                       // 000000001824: 818A0680
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001828: BF870481
	v_xor_b32_e32 v3, s10, v3                                  // 00000000182C: 3A06060A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001830: BF870091
	v_clz_i32_u32_e32 v4, v3                                   // 000000001834: 7E087303
	v_min_u32_e32 v4, 32, v4                                   // 000000001838: 260808A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 00000000183C: BF870131
	v_lshlrev_b32_e32 v6, 23, v4                               // 000000001840: 300C0897
	v_alignbit_b32 v1, s9, s7, v1                              // 000000001844: D6160001 04040E09
	v_sub_nc_u32_e32 v5, 31, v4                                // 00000000184C: 4C0A089F
	v_cndmask_b32_e64 v1, v1, s9, s8                           // 000000001850: D5010001 00201301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001858: BF8704B1
	v_alignbit_b32 v2, v2, v1, 30                              // 00000000185C: D6160002 027A0302
	v_alignbit_b32 v1, v1, s7, 30                              // 000000001864: D6160001 02780F01
	s_lshr_b32 s7, s3, 29                                      // 00000000186C: 85079D03
	s_lshl_b32 s7, s7, 31                                      // 000000001870: 84079F07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001874: BF870112
	v_xor_b32_e32 v2, s10, v2                                  // 000000001878: 3A04040A
	v_xor_b32_e32 v1, s10, v1                                  // 00000000187C: 3A02020A
	s_or_b32 s8, s7, 0.5                                       // 000000001880: 8C08F007
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001884: BF870199
	v_sub_nc_u32_e32 v6, s8, v6                                // 000000001888: 4C0C0C08
	v_alignbit_b32 v3, v3, v2, v5                              // 00000000188C: D6160003 04160503
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001894: BF870093
	v_alignbit_b32 v1, v2, v1, v5                              // 000000001898: D6160001 04160302
	v_alignbit_b32 v2, v3, v1, 9                               // 0000000018A0: D6160002 02260303
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018A8: BF870091
	v_clz_i32_u32_e32 v5, v2                                   // 0000000018AC: 7E0A7302
	v_min_u32_e32 v5, 32, v5                                   // 0000000018B0: 260A0AA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018B4: BF870091
	v_sub_nc_u32_e32 v7, 31, v5                                // 0000000018B8: 4C0E0A9F
	v_alignbit_b32 v1, v2, v1, v7                              // 0000000018BC: D6160001 041E0302
	v_lshrrev_b32_e32 v2, 9, v3                                // 0000000018C4: 32040689
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018C8: BF870112
	v_lshrrev_b32_e32 v1, 9, v1                                // 0000000018CC: 32020289
	v_or_b32_e32 v2, v2, v6                                    // 0000000018D0: 38040D02
	v_add_nc_u32_e32 v4, v5, v4                                // 0000000018D4: 4A080905
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D8: BF870091
	v_lshlrev_b32_e32 v3, 23, v4                               // 0000000018DC: 30060897
	v_sub_nc_u32_e32 v1, v1, v3                                // 0000000018E0: 4C020701
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018E4: BF870114
	v_mul_f32_e32 v3, 0x3fc90fda, v2                           // 0000000018E8: 100604FF 3FC90FDA
	v_add_nc_u32_e32 v1, 0x33000000, v1                        // 0000000018F0: 4A0202FF 33000000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018F8: BF870112
	v_fma_f32 v4, 0x3fc90fda, v2, -v3                          // 0000000018FC: D6130004 840E04FF 3FC90FDA
	v_or_b32_e32 v1, s7, v1                                    // 000000001908: 38020207
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000190C: BF8704A2
	v_fmac_f32_e32 v4, 0x33a22168, v2                          // 000000001910: 560804FF 33A22168
	s_lshr_b32 s7, s3, 30                                      // 000000001918: 85079E03
	s_add_i32 s6, s6, s7                                       // 00000000191C: 81060706
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001920: BF870091
	v_fmac_f32_e32 v4, 0x3fc90fda, v1                          // 000000001924: 560802FF 3FC90FDA
	v_add_f32_e32 v1, v3, v4                                   // 00000000192C: 06020903
	s_cbranch_execz 4                                          // 000000001930: BFA50004 <E_3n67+0x344>
	v_mov_b32_e32 v2, s6                                       // 000000001934: 7E040206
	s_branch 16                                                // 000000001938: BFA00010 <E_3n67+0x37c>
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 00000000193C: 916A037E
	s_cbranch_vccnz 65532                                      // 000000001940: BFA4FFFC <E_3n67+0x334>
	v_mul_f32_e64 v1, 0x3f22f983, s2                           // 000000001944: D5080001 000004FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001950: BF870091
	v_rndne_f32_e32 v2, v1                                     // 000000001954: 7E044701
	v_fma_f32 v1, 0xbfc90fda, v2, s2                           // 000000001958: D6130001 000A04FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001964: BF870091
	v_fmac_f32_e32 v1, 0xb3a22168, v2                          // 000000001968: 560204FF B3A22168
	v_fmac_f32_e32 v1, 0xa7c234c4, v2                          // 000000001970: 560204FF A7C234C4
	v_cvt_i32_f32_e32 v2, v2                                   // 000000001978: 7E041102
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000197C: BF870141
	v_dual_mul_f32 v3, v1, v1 :: v_dual_lshlrev_b32 v6, 30, v2 // 000000001980: C8E20301 0306049E
	s_mov_b32 s3, 0xb94c1982                                   // 000000001988: BE8300FF B94C1982
	s_mov_b32 s6, 0x37d75334                                   // 000000001990: BE8600FF 37D75334
	v_xor_b32_e32 v0, s2, v0                                   // 000000001998: 3A000002
	v_fmaak_f32 v4, s3, v3, 0x3c0881c4                         // 00000000199C: 5A080603 3C0881C4
	s_lshl_b64 s[4:5], s[4:5], 2                               // 0000000019A4: 84848204
	v_and_b32_e32 v2, 1, v2                                    // 0000000019A8: 36040481
	s_add_u32 s0, s0, s4                                       // 0000000019AC: 80000400
	s_addc_u32 s1, s1, s5                                      // 0000000019B0: 82010501
	v_fmaak_f32 v4, v3, v4, 0xbe2aaa9d                         // 0000000019B4: 5A080903 BE2AAA9D
	v_fmaak_f32 v5, s6, v3, 0xbab64f3b                         // 0000000019BC: 5A0A0606 BAB64F3B
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 0000000019C4: 7C940480
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000019C8: BF870193
	v_mul_f32_e32 v4, v3, v4                                   // 0000000019CC: 10080903
	v_fmaak_f32 v5, v3, v5, 0x3d2aabf7                         // 0000000019D0: 5A0A0B03 3D2AABF7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000019D8: BF870112
	v_dual_fmac_f32 v1, v1, v4 :: v_dual_and_b32 v6, 0x80000000, v6// 0000000019DC: C8240901 01060CFF 80000000
	v_fmaak_f32 v5, v3, v5, 0xbf000004                         // 0000000019E8: 5A0A0B03 BF000004
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000019F0: BF870112
	v_xor_b32_e32 v0, v0, v6                                   // 0000000019F4: 3A000D00
	v_fma_f32 v3, v3, v5, 1.0                                  // 0000000019F8: D6130003 03CA0B03
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001A00: BF870121
	v_cndmask_b32_e32 v1, v3, v1, vcc_lo                       // 000000001A04: 02020303
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 000000001A08: D47E006A 0001FE02 000001F8
	v_xor_b32_e32 v0, v0, v1                                   // 000000001A14: 3A000300
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A18: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_cndmask_b32 v0, 0x7fc00000, v0// 000000001A1C: CA120080 010000FF 7FC00000
	global_store_b32 v1, v0, s[0:1]                            // 000000001A28: DC6A0000 00000001
	s_nop 0                                                    // 000000001A30: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A34: BFB60003
	s_endpgm                                                   // 000000001A38: BFB00000
