
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n68>:
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
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001674: BF870121
	v_cmp_ngt_f32_e64 s2, 0x48000000, |v0|                     // 000000001678: D41B0202 000200FF 48000000
	v_readfirstlane_b32 s6, v0                                 // 000000001684: 7E0C0500
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001688: 8B6A027E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000168C: BF870001
	s_and_b32 s2, s6, 0x7fffffff                               // 000000001690: 8B02FF06 7FFFFFFF
	s_cbranch_vccz 166                                         // 000000001698: BFA300A6 <E_3n68+0x334>
	s_and_b32 s3, s2, 0x7fffff                                 // 00000000169C: 8B03FF02 007FFFFF
	s_lshr_b32 s6, s2, 23                                      // 0000000016A4: 85069702
	s_bitset1_b32 s3, 23                                       // 0000000016A8: BE831297
	s_addk_i32 s6, 0xff88                                      // 0000000016AC: B786FF88
	s_mul_hi_u32 s7, s3, 0xfe5163ab                            // 0000000016B0: 9687FF03 FE5163AB
	s_mul_i32 s8, s3, 0x3c439041                               // 0000000016B8: 9608FF03 3C439041
	s_mul_hi_u32 s9, s3, 0x3c439041                            // 0000000016C0: 9689FF03 3C439041
	s_add_u32 s7, s7, s8                                       // 0000000016C8: 80070807
	s_addc_u32 s8, 0, s9                                       // 0000000016CC: 82080980
	s_mul_i32 s9, s3, 0xdb629599                               // 0000000016D0: 9609FF03 DB629599
	s_mul_hi_u32 s10, s3, 0xdb629599                           // 0000000016D8: 968AFF03 DB629599
	s_add_u32 s8, s8, s9                                       // 0000000016E0: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000016E4: 82090A80
	s_mul_i32 s10, s3, 0xf534ddc0                              // 0000000016E8: 960AFF03 F534DDC0
	s_mul_hi_u32 s11, s3, 0xf534ddc0                           // 0000000016F0: 968BFF03 F534DDC0
	s_add_u32 s9, s9, s10                                      // 0000000016F8: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000016FC: 820A0B80
	s_mul_i32 s11, s3, 0xfc2757d1                              // 000000001700: 960BFF03 FC2757D1
	s_mul_hi_u32 s12, s3, 0xfc2757d1                           // 000000001708: 968CFF03 FC2757D1
	s_add_u32 s10, s10, s11                                    // 000000001710: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 000000001714: 820B0C80
	s_mul_i32 s12, s3, 0x4e441529                              // 000000001718: 960CFF03 4E441529
	s_mul_hi_u32 s13, s3, 0x4e441529                           // 000000001720: 968DFF03 4E441529
	s_add_u32 s11, s11, s12                                    // 000000001728: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 00000000172C: 820C0D80
	s_cmp_gt_u32 s6, 63                                        // 000000001730: BF08BF06
	s_mul_i32 s13, s3, 0xfe5163ab                              // 000000001734: 960DFF03 FE5163AB
	s_mul_hi_u32 s14, s3, 0xa2f9836e                           // 00000000173C: 968EFF03 A2F9836E
	s_mul_i32 s3, s3, 0xa2f9836e                               // 000000001744: 9603FF03 A2F9836E
	s_cselect_b32 s15, s8, s10                                 // 00000000174C: 980F0A08
	s_cselect_b32 s7, s7, s9                                   // 000000001750: 98070907
	s_cselect_b32 s8, s13, s8                                  // 000000001754: 9808080D
	s_add_u32 s3, s12, s3                                      // 000000001758: 8003030C
	s_addc_u32 s12, 0, s14                                     // 00000000175C: 820C0E80
	s_cmp_gt_u32 s6, 63                                        // 000000001760: BF08BF06
	s_cselect_b32 s13, 0xffffffc0, 0                           // 000000001764: 980D80FF FFFFFFC0
	s_cselect_b32 s9, s9, s11                                  // 00000000176C: 98090B09
	s_cselect_b32 s3, s10, s3                                  // 000000001770: 9803030A
	s_cselect_b32 s10, s11, s12                                // 000000001774: 980A0C0B
	s_add_i32 s13, s13, s6                                     // 000000001778: 810D060D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000177C: BF870009
	s_cmp_gt_u32 s13, 31                                       // 000000001780: BF089F0D
	s_cselect_b32 s6, 0xffffffe0, 0                            // 000000001784: 980680FF FFFFFFE0
	s_cselect_b32 s11, s9, s3                                  // 00000000178C: 980B0309
	s_cselect_b32 s3, s3, s10                                  // 000000001790: 98030A03
	s_cselect_b32 s9, s15, s9                                  // 000000001794: 9809090F
	s_cselect_b32 s10, s7, s15                                 // 000000001798: 980A0F07
	s_cselect_b32 s7, s8, s7                                   // 00000000179C: 98070708
	s_add_i32 s6, s6, s13                                      // 0000000017A0: 81060D06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A4: BF870009
	s_cmp_gt_u32 s6, 31                                        // 0000000017A8: BF089F06
	s_cselect_b32 s8, 0xffffffe0, 0                            // 0000000017AC: 980880FF FFFFFFE0
	s_cselect_b32 s3, s11, s3                                  // 0000000017B4: 9803030B
	s_cselect_b32 s11, s9, s11                                 // 0000000017B8: 980B0B09
	s_cselect_b32 s9, s10, s9                                  // 0000000017BC: 9809090A
	s_cselect_b32 s7, s7, s10                                  // 0000000017C0: 98070A07
	s_add_i32 s8, s8, s6                                       // 0000000017C4: 81080608
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017C8: BF8700C9
	s_sub_i32 s6, 32, s8                                       // 0000000017CC: 818608A0
	s_cmp_eq_u32 s8, 0                                         // 0000000017D0: BF068008
	v_mov_b32_e32 v1, s6                                       // 0000000017D4: 7E020206
	s_cselect_b32 s8, -1, 0                                    // 0000000017D8: 980880C1
	v_alignbit_b32 v2, s3, s11, v1                             // 0000000017DC: D6160002 04041603
	v_alignbit_b32 v3, s11, s9, v1                             // 0000000017E4: D6160003 0404120B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017EC: BF870112
	v_readfirstlane_b32 s6, v2                                 // 0000000017F0: 7E0C0502
	v_cndmask_b32_e64 v2, v3, s11, s8                          // 0000000017F4: D5010002 00201703
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017FC: BF870002
	s_cselect_b32 s3, s3, s6                                   // 000000001800: 98030603
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001804: BF870481
	v_alignbit_b32 v3, s3, v2, 30                              // 000000001808: D6160003 027A0403
	s_bfe_u32 s6, s3, 0x1001d                                  // 000000001810: 9306FF03 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001818: BF870009
	s_sub_i32 s10, 0, s6                                       // 00000000181C: 818A0680
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001820: BF870481
	v_xor_b32_e32 v3, s10, v3                                  // 000000001824: 3A06060A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001828: BF870091
	v_clz_i32_u32_e32 v4, v3                                   // 00000000182C: 7E087303
	v_min_u32_e32 v4, 32, v4                                   // 000000001830: 260808A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001834: BF870131
	v_lshlrev_b32_e32 v6, 23, v4                               // 000000001838: 300C0897
	v_alignbit_b32 v1, s9, s7, v1                              // 00000000183C: D6160001 04040E09
	v_sub_nc_u32_e32 v5, 31, v4                                // 000000001844: 4C0A089F
	v_cndmask_b32_e64 v1, v1, s9, s8                           // 000000001848: D5010001 00201301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001850: BF8704B1
	v_alignbit_b32 v2, v2, v1, 30                              // 000000001854: D6160002 027A0302
	v_alignbit_b32 v1, v1, s7, 30                              // 00000000185C: D6160001 02780F01
	s_lshr_b32 s7, s3, 29                                      // 000000001864: 85079D03
	s_lshl_b32 s7, s7, 31                                      // 000000001868: 84079F07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000186C: BF870112
	v_xor_b32_e32 v2, s10, v2                                  // 000000001870: 3A04040A
	v_xor_b32_e32 v1, s10, v1                                  // 000000001874: 3A02020A
	s_or_b32 s8, s7, 0.5                                       // 000000001878: 8C08F007
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000187C: BF870199
	v_sub_nc_u32_e32 v6, s8, v6                                // 000000001880: 4C0C0C08
	v_alignbit_b32 v3, v3, v2, v5                              // 000000001884: D6160003 04160503
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000188C: BF870093
	v_alignbit_b32 v1, v2, v1, v5                              // 000000001890: D6160001 04160302
	v_alignbit_b32 v2, v3, v1, 9                               // 000000001898: D6160002 02260303
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018A0: BF870091
	v_clz_i32_u32_e32 v5, v2                                   // 0000000018A4: 7E0A7302
	v_min_u32_e32 v5, 32, v5                                   // 0000000018A8: 260A0AA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018AC: BF870091
	v_sub_nc_u32_e32 v7, 31, v5                                // 0000000018B0: 4C0E0A9F
	v_alignbit_b32 v1, v2, v1, v7                              // 0000000018B4: D6160001 041E0302
	v_lshrrev_b32_e32 v2, 9, v3                                // 0000000018BC: 32040689
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018C0: BF870112
	v_lshrrev_b32_e32 v1, 9, v1                                // 0000000018C4: 32020289
	v_or_b32_e32 v2, v2, v6                                    // 0000000018C8: 38040D02
	v_add_nc_u32_e32 v4, v5, v4                                // 0000000018CC: 4A080905
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D0: BF870091
	v_lshlrev_b32_e32 v3, 23, v4                               // 0000000018D4: 30060897
	v_sub_nc_u32_e32 v1, v1, v3                                // 0000000018D8: 4C020701
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018DC: BF870114
	v_mul_f32_e32 v3, 0x3fc90fda, v2                           // 0000000018E0: 100604FF 3FC90FDA
	v_add_nc_u32_e32 v1, 0x33000000, v1                        // 0000000018E8: 4A0202FF 33000000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018F0: BF870112
	v_fma_f32 v4, 0x3fc90fda, v2, -v3                          // 0000000018F4: D6130004 840E04FF 3FC90FDA
	v_or_b32_e32 v1, s7, v1                                    // 000000001900: 38020207
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001904: BF8704A2
	v_fmac_f32_e32 v4, 0x33a22168, v2                          // 000000001908: 560804FF 33A22168
	s_lshr_b32 s7, s3, 30                                      // 000000001910: 85079E03
	s_add_i32 s6, s6, s7                                       // 000000001914: 81060706
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001918: BF870091
	v_fmac_f32_e32 v4, 0x3fc90fda, v1                          // 00000000191C: 560802FF 3FC90FDA
	v_add_f32_e32 v1, v3, v4                                   // 000000001924: 06020903
	s_cbranch_execz 4                                          // 000000001928: BFA50004 <E_3n68+0x33c>
	v_mov_b32_e32 v2, s6                                       // 00000000192C: 7E040206
	s_branch 16                                                // 000000001930: BFA00010 <E_3n68+0x374>
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 000000001934: 916A037E
	s_cbranch_vccnz 65532                                      // 000000001938: BFA4FFFC <E_3n68+0x32c>
	v_mul_f32_e64 v1, 0x3f22f983, s2                           // 00000000193C: D5080001 000004FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001948: BF870091
	v_rndne_f32_e32 v2, v1                                     // 00000000194C: 7E044701
	v_fma_f32 v1, 0xbfc90fda, v2, s2                           // 000000001950: D6130001 000A04FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000195C: BF870091
	v_fmac_f32_e32 v1, 0xb3a22168, v2                          // 000000001960: 560204FF B3A22168
	v_fmac_f32_e32 v1, 0xa7c234c4, v2                          // 000000001968: 560204FF A7C234C4
	v_cvt_i32_f32_e32 v2, v2                                   // 000000001970: 7E041102
	v_sub_f32_e32 v3, 0x3fc90fdb, v0                           // 000000001974: 080600FF 3FC90FDB
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000197C: BF870121
	v_cmp_ngt_f32_e64 s3, 0x48000000, |v3|                     // 000000001980: D41B0203 000206FF 48000000
	v_readfirstlane_b32 s6, v3                                 // 00000000198C: 7E0C0503
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001990: 8B6A037E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001994: BF870001
	s_and_b32 s3, s6, 0x7fffffff                               // 000000001998: 8B03FF06 7FFFFFFF
	s_cbranch_vccz 164                                         // 0000000019A0: BFA300A4 <E_3n68+0x634>
	s_and_b32 s6, s3, 0x7fffff                                 // 0000000019A4: 8B06FF03 007FFFFF
	s_lshr_b32 s7, s3, 23                                      // 0000000019AC: 85079703
	s_bitset1_b32 s6, 23                                       // 0000000019B0: BE861297
	s_addk_i32 s7, 0xff88                                      // 0000000019B4: B787FF88
	s_mul_hi_u32 s8, s6, 0xfe5163ab                            // 0000000019B8: 9688FF06 FE5163AB
	s_mul_i32 s9, s6, 0x3c439041                               // 0000000019C0: 9609FF06 3C439041
	s_mul_hi_u32 s10, s6, 0x3c439041                           // 0000000019C8: 968AFF06 3C439041
	s_add_u32 s8, s8, s9                                       // 0000000019D0: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000019D4: 82090A80
	s_mul_i32 s10, s6, 0xdb629599                              // 0000000019D8: 960AFF06 DB629599
	s_mul_hi_u32 s11, s6, 0xdb629599                           // 0000000019E0: 968BFF06 DB629599
	s_add_u32 s9, s9, s10                                      // 0000000019E8: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000019EC: 820A0B80
	s_mul_i32 s11, s6, 0xf534ddc0                              // 0000000019F0: 960BFF06 F534DDC0
	s_mul_hi_u32 s12, s6, 0xf534ddc0                           // 0000000019F8: 968CFF06 F534DDC0
	s_add_u32 s10, s10, s11                                    // 000000001A00: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 000000001A04: 820B0C80
	s_mul_i32 s12, s6, 0xfc2757d1                              // 000000001A08: 960CFF06 FC2757D1
	s_mul_hi_u32 s13, s6, 0xfc2757d1                           // 000000001A10: 968DFF06 FC2757D1
	s_add_u32 s11, s11, s12                                    // 000000001A18: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 000000001A1C: 820C0D80
	s_mul_i32 s13, s6, 0x4e441529                              // 000000001A20: 960DFF06 4E441529
	s_mul_hi_u32 s14, s6, 0x4e441529                           // 000000001A28: 968EFF06 4E441529
	s_add_u32 s12, s12, s13                                    // 000000001A30: 800C0D0C
	s_addc_u32 s13, 0, s14                                     // 000000001A34: 820D0E80
	s_cmp_gt_u32 s7, 63                                        // 000000001A38: BF08BF07
	s_mul_i32 s14, s6, 0xfe5163ab                              // 000000001A3C: 960EFF06 FE5163AB
	s_mul_hi_u32 s15, s6, 0xa2f9836e                           // 000000001A44: 968FFF06 A2F9836E
	s_mul_i32 s6, s6, 0xa2f9836e                               // 000000001A4C: 9606FF06 A2F9836E
	s_cselect_b32 s16, s9, s11                                 // 000000001A54: 98100B09
	s_cselect_b32 s8, s8, s10                                  // 000000001A58: 98080A08
	s_cselect_b32 s9, s14, s9                                  // 000000001A5C: 9809090E
	s_add_u32 s6, s13, s6                                      // 000000001A60: 8006060D
	s_addc_u32 s13, 0, s15                                     // 000000001A64: 820D0F80
	s_cmp_gt_u32 s7, 63                                        // 000000001A68: BF08BF07
	s_cselect_b32 s14, 0xffffffc0, 0                           // 000000001A6C: 980E80FF FFFFFFC0
	s_cselect_b32 s10, s10, s12                                // 000000001A74: 980A0C0A
	s_cselect_b32 s6, s11, s6                                  // 000000001A78: 9806060B
	s_cselect_b32 s11, s12, s13                                // 000000001A7C: 980B0D0C
	s_add_i32 s14, s14, s7                                     // 000000001A80: 810E070E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A84: BF870009
	s_cmp_gt_u32 s14, 31                                       // 000000001A88: BF089F0E
	s_cselect_b32 s7, 0xffffffe0, 0                            // 000000001A8C: 980780FF FFFFFFE0
	s_cselect_b32 s12, s10, s6                                 // 000000001A94: 980C060A
	s_cselect_b32 s6, s6, s11                                  // 000000001A98: 98060B06
	s_cselect_b32 s10, s16, s10                                // 000000001A9C: 980A0A10
	s_cselect_b32 s11, s8, s16                                 // 000000001AA0: 980B1008
	s_cselect_b32 s8, s9, s8                                   // 000000001AA4: 98080809
	s_add_i32 s7, s7, s14                                      // 000000001AA8: 81070E07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001AAC: BF870009
	s_cmp_gt_u32 s7, 31                                        // 000000001AB0: BF089F07
	s_cselect_b32 s9, 0xffffffe0, 0                            // 000000001AB4: 980980FF FFFFFFE0
	s_cselect_b32 s6, s12, s6                                  // 000000001ABC: 9806060C
	s_cselect_b32 s12, s10, s12                                // 000000001AC0: 980C0C0A
	s_cselect_b32 s10, s11, s10                                // 000000001AC4: 980A0A0B
	s_cselect_b32 s8, s8, s11                                  // 000000001AC8: 98080B08
	s_add_i32 s9, s9, s7                                       // 000000001ACC: 81090709
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001AD0: BF8700C9
	s_sub_i32 s7, 32, s9                                       // 000000001AD4: 818709A0
	s_cmp_eq_u32 s9, 0                                         // 000000001AD8: BF068009
	v_mov_b32_e32 v4, s7                                       // 000000001ADC: 7E080207
	s_cselect_b32 s9, -1, 0                                    // 000000001AE0: 980980C1
	v_alignbit_b32 v5, s6, s12, v4                             // 000000001AE4: D6160005 04101806
	v_alignbit_b32 v6, s12, s10, v4                            // 000000001AEC: D6160006 0410140C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001AF4: BF870112
	v_readfirstlane_b32 s7, v5                                 // 000000001AF8: 7E0E0505
	v_cndmask_b32_e64 v5, v6, s12, s9                          // 000000001AFC: D5010005 00241906
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B04: BF870002
	s_cselect_b32 s6, s6, s7                                   // 000000001B08: 98060706
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001B0C: BF870481
	v_alignbit_b32 v6, s6, v5, 30                              // 000000001B10: D6160006 027A0A06
	s_bfe_u32 s11, s6, 0x1001d                                 // 000000001B18: 930BFF06 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B20: BF870009
	s_sub_i32 s7, 0, s11                                       // 000000001B24: 81870B80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001B28: BF870481
	v_xor_b32_e32 v6, s7, v6                                   // 000000001B2C: 3A0C0C07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B30: BF870091
	v_clz_i32_u32_e32 v7, v6                                   // 000000001B34: 7E0E7306
	v_min_u32_e32 v7, 32, v7                                   // 000000001B38: 260E0EA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001B3C: BF870131
	v_lshlrev_b32_e32 v9, 23, v7                               // 000000001B40: 30120E97
	v_alignbit_b32 v4, s10, s8, v4                             // 000000001B44: D6160004 0410100A
	v_sub_nc_u32_e32 v8, 31, v7                                // 000000001B4C: 4C100E9F
	v_cndmask_b32_e64 v4, v4, s10, s9                          // 000000001B50: D5010004 00241504
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001B58: BF870121
	v_alignbit_b32 v5, v5, v4, 30                              // 000000001B5C: D6160005 027A0905
	v_alignbit_b32 v4, v4, s8, 30                              // 000000001B64: D6160004 02781104
	v_xor_b32_e32 v5, s7, v5                                   // 000000001B6C: 3A0A0A07
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B70: BF870002
	v_xor_b32_e32 v4, s7, v4                                   // 000000001B74: 3A080807
	s_lshr_b32 s7, s6, 29                                      // 000000001B78: 85079D06
	s_lshr_b32 s6, s6, 30                                      // 000000001B7C: 85069E06
	s_lshl_b32 s7, s7, 31                                      // 000000001B80: 84079F07
	v_alignbit_b32 v6, v6, v5, v8                              // 000000001B84: D6160006 04220B06
	v_alignbit_b32 v4, v5, v4, v8                              // 000000001B8C: D6160004 04220905
	s_or_b32 s8, s7, 0.5                                       // 000000001B94: 8C08F007
	s_add_i32 s6, s11, s6                                      // 000000001B98: 8106060B
	v_sub_nc_u32_e32 v9, s8, v9                                // 000000001B9C: 4C121208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BA0: BF870092
	v_alignbit_b32 v5, v6, v4, 9                               // 000000001BA4: D6160005 02260906
	v_clz_i32_u32_e32 v8, v5                                   // 000000001BAC: 7E107305
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BB0: BF870091
	v_min_u32_e32 v8, 32, v8                                   // 000000001BB4: 261010A0
	v_sub_nc_u32_e32 v10, 31, v8                               // 000000001BB8: 4C14109F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001BBC: BF870121
	v_alignbit_b32 v4, v5, v4, v10                             // 000000001BC0: D6160004 042A0905
	v_lshrrev_b32_e32 v5, 9, v6                                // 000000001BC8: 320A0C89
	v_lshrrev_b32_e32 v4, 9, v4                                // 000000001BCC: 32080889
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001BD0: BF8700A2
	v_or_b32_e32 v5, v5, v9                                    // 000000001BD4: 380A1305
	v_add_nc_u32_e32 v7, v8, v7                                // 000000001BD8: 4A0E0F08
	v_lshlrev_b32_e32 v6, 23, v7                               // 000000001BDC: 300C0E97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001BE0: BF870211
	v_sub_nc_u32_e32 v4, v4, v6                                // 000000001BE4: 4C080D04
	v_mul_f32_e32 v6, 0x3fc90fda, v5                           // 000000001BE8: 100C0AFF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001BF0: BF870112
	v_add_nc_u32_e32 v4, 0x33000000, v4                        // 000000001BF4: 4A0808FF 33000000
	v_fma_f32 v7, 0x3fc90fda, v5, -v6                          // 000000001BFC: D6130007 841A0AFF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001C08: BF870112
	v_or_b32_e32 v4, s7, v4                                    // 000000001C0C: 38080807
	v_fmac_f32_e32 v7, 0x33a22168, v5                          // 000000001C10: 560E0AFF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C18: BF870091
	v_fmac_f32_e32 v7, 0x3fc90fda, v4                          // 000000001C1C: 560E08FF 3FC90FDA
	v_add_f32_e32 v4, v6, v7                                   // 000000001C24: 06080F06
	s_cbranch_execz 2                                          // 000000001C28: BFA50002 <E_3n68+0x634>
	v_mov_b32_e32 v5, s6                                       // 000000001C2C: 7E0A0206
	s_branch 14                                                // 000000001C30: BFA0000E <E_3n68+0x66c>
	v_mul_f32_e64 v4, 0x3f22f983, s3                           // 000000001C34: D5080004 000006FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C40: BF870091
	v_rndne_f32_e32 v5, v4                                     // 000000001C44: 7E0A4704
	v_fma_f32 v4, 0xbfc90fda, v5, s3                           // 000000001C48: D6130004 000E0AFF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C54: BF870091
	v_fmac_f32_e32 v4, 0xb3a22168, v5                          // 000000001C58: 56080AFF B3A22168
	v_fmac_f32_e32 v4, 0xa7c234c4, v5                          // 000000001C60: 56080AFF A7C234C4
	v_cvt_i32_f32_e32 v5, v5                                   // 000000001C68: 7E0A1105
	v_dual_mul_f32 v6, v1, v1 :: v_dual_lshlrev_b32 v7, 30, v2 // 000000001C6C: C8E20301 0606049E
	s_mov_b32 s6, 0xb94c1982                                   // 000000001C74: BE8600FF B94C1982
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001C7C: BF870123
	v_mul_f32_e32 v8, v4, v4                                   // 000000001C80: 10100904
	s_mov_b32 s7, 0x37d75334                                   // 000000001C84: BE8700FF 37D75334
	v_fmaak_f32 v9, s6, v6, 0x3c0881c4                         // 000000001C8C: 5A120C06 3C0881C4
	v_xor_b32_e32 v0, s2, v0                                   // 000000001C94: 3A000002
	v_and_b32_e32 v7, 0x80000000, v7                           // 000000001C98: 360E0EFF 80000000
	v_dual_fmaak_f32 v11, s6, v8, 0x3c0881c4 :: v_dual_and_b32 v2, 1, v2// 000000001CA0: C8641006 0B020481 3C0881C4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001CAC: BF870224
	v_fmaak_f32 v9, v6, v9, 0xbe2aaa9d                         // 000000001CB0: 5A121306 BE2AAA9D
	v_fmaak_f32 v10, s7, v6, 0xbab64f3b                        // 000000001CB8: 5A140C07 BAB64F3B
	v_xor_b32_e32 v0, v0, v7                                   // 000000001CC0: 3A000F00
	v_xor_b32_e32 v3, s3, v3                                   // 000000001CC4: 3A060603
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 000000001CC8: 7C940480
	v_mul_f32_e32 v9, v6, v9                                   // 000000001CCC: 10121306
	v_fmaak_f32 v7, v6, v10, 0x3d2aabf7                        // 000000001CD0: 5A0E1506 3D2AABF7
	v_fmaak_f32 v10, v8, v11, 0xbe2aaa9d                       // 000000001CD8: 5A141708 BE2AAA9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001CE0: BF870113
	v_dual_fmaak_f32 v12, s7, v8, 0xbab64f3b :: v_dual_fmac_f32 v1, v1, v9// 000000001CE4: C8401007 0C001301 BAB64F3B
	v_dual_mul_f32 v10, v8, v10 :: v_dual_fmaak_f32 v7, v6, v7, 0xbf000004// 000000001CF0: C8C21508 0A060F06 BF000004
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001CFC: BF870112
	v_fmaak_f32 v11, v8, v12, 0x3d2aabf7                       // 000000001D00: 5A161908 3D2AABF7
	v_fmac_f32_e32 v4, v4, v10                                 // 000000001D08: 56081504
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D0C: BF870093
	v_fma_f32 v6, v6, v7, 1.0                                  // 000000001D10: D6130006 03CA0F06
	v_cndmask_b32_e32 v1, v6, v1, vcc_lo                       // 000000001D18: 02020306
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001D1C: BF8700B1
	v_xor_b32_e32 v0, v0, v1                                   // 000000001D20: 3A000300
	v_lshlrev_b32_e32 v9, 30, v5                               // 000000001D24: 30120A9E
	v_and_b32_e32 v5, 1, v5                                    // 000000001D28: 360A0A81
	v_cmp_eq_u32_e32 vcc_lo, 0, v5                             // 000000001D2C: 7C940A80
	v_fmaak_f32 v11, v8, v11, 0xbf000004                       // 000000001D30: 5A161708 BF000004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D38: BF870091
	v_fma_f32 v7, v8, v11, 1.0                                 // 000000001D3C: D6130007 03CA1708
	v_cndmask_b32_e32 v2, v7, v4, vcc_lo                       // 000000001D44: 02040907
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 000000001D48: D47E006A 0001FE02 000001F8
	v_cndmask_b32_e32 v0, 0x7fc00000, v0, vcc_lo               // 000000001D54: 020000FF 7FC00000
	v_and_b32_e32 v8, 0x80000000, v9                           // 000000001D5C: 361012FF 80000000
	v_cmp_class_f32_e64 vcc_lo, s3, 0x1f8                      // 000000001D64: D47E006A 0001FE03 000001F8
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001D70: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001D74: BF870119
	s_add_u32 s0, s0, s2                                       // 000000001D78: 80000200
	v_xor_b32_e32 v3, v3, v8                                   // 000000001D7C: 3A061103
	s_addc_u32 s1, s1, s3                                      // 000000001D80: 82010301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D84: BF870091
	v_xor_b32_e32 v1, v3, v2                                   // 000000001D88: 3A020503
	v_cndmask_b32_e32 v1, 0x7fc00000, v1, vcc_lo               // 000000001D8C: 020202FF 7FC00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001D94: BF870121
	v_div_scale_f32 v2, null, v1, v1, v0                       // 000000001D98: D6FC7C02 04020301
	v_div_scale_f32 v5, vcc_lo, v0, v1, v0                     // 000000001DA0: D6FC6A05 04020300
	v_rcp_f32_e32 v3, v2                                       // 000000001DA8: 7E065502
	s_waitcnt_depctr 0xfff                                     // 000000001DAC: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 000000001DB0: D6130004 23CA0702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DB8: BF870091
	v_fmac_f32_e32 v3, v4, v3                                  // 000000001DBC: 56060704
	v_mul_f32_e32 v4, v5, v3                                   // 000000001DC0: 10080705
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DC4: BF870091
	v_fma_f32 v6, -v2, v4, v5                                  // 000000001DC8: D6130006 24160902
	v_fmac_f32_e32 v4, v6, v3                                  // 000000001DD0: 56080706
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DD4: BF870091
	v_fma_f32 v2, -v2, v4, v5                                  // 000000001DD8: D6130002 24160902
	v_div_fmas_f32 v2, v2, v3, v4                              // 000000001DE0: D6370002 04120702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001DE8: BF870001
	v_div_fixup_f32 v0, v2, v1, v0                             // 000000001DEC: D6270000 04020302
	v_mov_b32_e32 v1, 0                                        // 000000001DF4: 7E020280
	global_store_b32 v1, v0, s[0:1]                            // 000000001DF8: DC6A0000 00000001
	s_nop 0                                                    // 000000001E00: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001E04: BFB60003
	s_endpgm                                                   // 000000001E08: BFB00000
