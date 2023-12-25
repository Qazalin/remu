
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n6>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_mov_b32 s4, s15                                          // 000000001610: BE84000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s2, s2, s15                                      // 000000001618: 80020F02
	s_addc_u32 s3, s3, s5                                      // 00000000161C: 82030503
	global_load_i8 v0, v0, s[2:3]                              // 000000001620: DC460000 00020000
	s_waitcnt vmcnt(0)                                         // 000000001628: BF8903F7
	v_cvt_f32_i32_e32 v0, v0                                   // 00000000162C: 7E000B00
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001630: BF870121
	v_cmp_ngt_f32_e64 s2, 0x48000000, |v0|                     // 000000001634: D41B0202 000200FF 48000000
	v_readfirstlane_b32 s3, v0                                 // 000000001640: 7E060500
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001644: 8B6A027E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001648: BF870001
	s_and_b32 s2, s3, 0x7fffffff                               // 00000000164C: 8B02FF03 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001654: BFA300A4 <E_3n6+0x2e8>
	s_and_b32 s3, s2, 0x7fffff                                 // 000000001658: 8B03FF02 007FFFFF
	s_lshr_b32 s6, s2, 23                                      // 000000001660: 85069702
	s_bitset1_b32 s3, 23                                       // 000000001664: BE831297
	s_addk_i32 s6, 0xff88                                      // 000000001668: B786FF88
	s_mul_hi_u32 s7, s3, 0xfe5163ab                            // 00000000166C: 9687FF03 FE5163AB
	s_mul_i32 s8, s3, 0x3c439041                               // 000000001674: 9608FF03 3C439041
	s_mul_hi_u32 s9, s3, 0x3c439041                            // 00000000167C: 9689FF03 3C439041
	s_add_u32 s7, s7, s8                                       // 000000001684: 80070807
	s_addc_u32 s8, 0, s9                                       // 000000001688: 82080980
	s_mul_i32 s9, s3, 0xdb629599                               // 00000000168C: 9609FF03 DB629599
	s_mul_hi_u32 s10, s3, 0xdb629599                           // 000000001694: 968AFF03 DB629599
	s_add_u32 s8, s8, s9                                       // 00000000169C: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000016A0: 82090A80
	s_mul_i32 s10, s3, 0xf534ddc0                              // 0000000016A4: 960AFF03 F534DDC0
	s_mul_hi_u32 s11, s3, 0xf534ddc0                           // 0000000016AC: 968BFF03 F534DDC0
	s_add_u32 s9, s9, s10                                      // 0000000016B4: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000016B8: 820A0B80
	s_mul_i32 s11, s3, 0xfc2757d1                              // 0000000016BC: 960BFF03 FC2757D1
	s_mul_hi_u32 s12, s3, 0xfc2757d1                           // 0000000016C4: 968CFF03 FC2757D1
	s_add_u32 s10, s10, s11                                    // 0000000016CC: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000016D0: 820B0C80
	s_mul_i32 s12, s3, 0x4e441529                              // 0000000016D4: 960CFF03 4E441529
	s_mul_hi_u32 s13, s3, 0x4e441529                           // 0000000016DC: 968DFF03 4E441529
	s_add_u32 s11, s11, s12                                    // 0000000016E4: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000016E8: 820C0D80
	s_cmp_gt_u32 s6, 63                                        // 0000000016EC: BF08BF06
	s_mul_i32 s13, s3, 0xfe5163ab                              // 0000000016F0: 960DFF03 FE5163AB
	s_mul_hi_u32 s14, s3, 0xa2f9836e                           // 0000000016F8: 968EFF03 A2F9836E
	s_mul_i32 s3, s3, 0xa2f9836e                               // 000000001700: 9603FF03 A2F9836E
	s_cselect_b32 s15, s8, s10                                 // 000000001708: 980F0A08
	s_cselect_b32 s7, s7, s9                                   // 00000000170C: 98070907
	s_cselect_b32 s8, s13, s8                                  // 000000001710: 9808080D
	s_add_u32 s3, s12, s3                                      // 000000001714: 8003030C
	s_addc_u32 s12, 0, s14                                     // 000000001718: 820C0E80
	s_cmp_gt_u32 s6, 63                                        // 00000000171C: BF08BF06
	s_cselect_b32 s13, 0xffffffc0, 0                           // 000000001720: 980D80FF FFFFFFC0
	s_cselect_b32 s9, s9, s11                                  // 000000001728: 98090B09
	s_cselect_b32 s3, s10, s3                                  // 00000000172C: 9803030A
	s_cselect_b32 s10, s11, s12                                // 000000001730: 980A0C0B
	s_add_i32 s13, s13, s6                                     // 000000001734: 810D060D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001738: BF870009
	s_cmp_gt_u32 s13, 31                                       // 00000000173C: BF089F0D
	s_cselect_b32 s6, 0xffffffe0, 0                            // 000000001740: 980680FF FFFFFFE0
	s_cselect_b32 s11, s9, s3                                  // 000000001748: 980B0309
	s_cselect_b32 s3, s3, s10                                  // 00000000174C: 98030A03
	s_cselect_b32 s9, s15, s9                                  // 000000001750: 9809090F
	s_cselect_b32 s10, s7, s15                                 // 000000001754: 980A0F07
	s_cselect_b32 s7, s8, s7                                   // 000000001758: 98070708
	s_add_i32 s6, s6, s13                                      // 00000000175C: 81060D06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001760: BF870009
	s_cmp_gt_u32 s6, 31                                        // 000000001764: BF089F06
	s_cselect_b32 s8, 0xffffffe0, 0                            // 000000001768: 980880FF FFFFFFE0
	s_cselect_b32 s3, s11, s3                                  // 000000001770: 9803030B
	s_cselect_b32 s11, s9, s11                                 // 000000001774: 980B0B09
	s_cselect_b32 s9, s10, s9                                  // 000000001778: 9809090A
	s_cselect_b32 s7, s7, s10                                  // 00000000177C: 98070A07
	s_add_i32 s8, s8, s6                                       // 000000001780: 81080608
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001784: BF8700C9
	s_sub_i32 s6, 32, s8                                       // 000000001788: 818608A0
	s_cmp_eq_u32 s8, 0                                         // 00000000178C: BF068008
	v_mov_b32_e32 v1, s6                                       // 000000001790: 7E020206
	s_cselect_b32 s8, -1, 0                                    // 000000001794: 980880C1
	v_alignbit_b32 v2, s3, s11, v1                             // 000000001798: D6160002 04041603
	v_alignbit_b32 v3, s11, s9, v1                             // 0000000017A0: D6160003 0404120B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017A8: BF870112
	v_readfirstlane_b32 s6, v2                                 // 0000000017AC: 7E0C0502
	v_cndmask_b32_e64 v2, v3, s11, s8                          // 0000000017B0: D5010002 00201703
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017B8: BF870002
	s_cselect_b32 s3, s3, s6                                   // 0000000017BC: 98030603
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017C0: BF870481
	v_alignbit_b32 v3, s3, v2, 30                              // 0000000017C4: D6160003 027A0403
	s_bfe_u32 s10, s3, 0x1001d                                 // 0000000017CC: 930AFF03 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017D4: BF870009
	s_sub_i32 s6, 0, s10                                       // 0000000017D8: 81860A80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017DC: BF870481
	v_xor_b32_e32 v3, s6, v3                                   // 0000000017E0: 3A060606
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E4: BF870091
	v_clz_i32_u32_e32 v4, v3                                   // 0000000017E8: 7E087303
	v_min_u32_e32 v4, 32, v4                                   // 0000000017EC: 260808A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000017F0: BF870131
	v_lshlrev_b32_e32 v6, 23, v4                               // 0000000017F4: 300C0897
	v_alignbit_b32 v1, s9, s7, v1                              // 0000000017F8: D6160001 04040E09
	v_sub_nc_u32_e32 v5, 31, v4                                // 000000001800: 4C0A089F
	v_cndmask_b32_e64 v1, v1, s9, s8                           // 000000001804: D5010001 00201301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000180C: BF870121
	v_alignbit_b32 v2, v2, v1, 30                              // 000000001810: D6160002 027A0302
	v_alignbit_b32 v1, v1, s7, 30                              // 000000001818: D6160001 02780F01
	v_xor_b32_e32 v2, s6, v2                                   // 000000001820: 3A040406
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001824: BF870002
	v_xor_b32_e32 v1, s6, v1                                   // 000000001828: 3A020206
	s_lshr_b32 s6, s3, 29                                      // 00000000182C: 85069D03
	s_lshr_b32 s3, s3, 30                                      // 000000001830: 85039E03
	s_lshl_b32 s6, s6, 31                                      // 000000001834: 84069F06
	v_alignbit_b32 v3, v3, v2, v5                              // 000000001838: D6160003 04160503
	v_alignbit_b32 v1, v2, v1, v5                              // 000000001840: D6160001 04160302
	s_or_b32 s7, s6, 0.5                                       // 000000001848: 8C07F006
	s_add_i32 s3, s10, s3                                      // 00000000184C: 8103030A
	v_sub_nc_u32_e32 v6, s7, v6                                // 000000001850: 4C0C0C07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001854: BF870092
	v_alignbit_b32 v2, v3, v1, 9                               // 000000001858: D6160002 02260303
	v_clz_i32_u32_e32 v5, v2                                   // 000000001860: 7E0A7302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001864: BF870091
	v_min_u32_e32 v5, 32, v5                                   // 000000001868: 260A0AA0
	v_sub_nc_u32_e32 v7, 31, v5                                // 00000000186C: 4C0E0A9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001870: BF870121
	v_alignbit_b32 v1, v2, v1, v7                              // 000000001874: D6160001 041E0302
	v_lshrrev_b32_e32 v2, 9, v3                                // 00000000187C: 32040689
	v_lshrrev_b32_e32 v1, 9, v1                                // 000000001880: 32020289
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001884: BF8700A2
	v_or_b32_e32 v2, v2, v6                                    // 000000001888: 38040D02
	v_add_nc_u32_e32 v4, v5, v4                                // 00000000188C: 4A080905
	v_lshlrev_b32_e32 v3, 23, v4                               // 000000001890: 30060897
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001894: BF870211
	v_sub_nc_u32_e32 v1, v1, v3                                // 000000001898: 4C020701
	v_mul_f32_e32 v3, 0x3fc90fda, v2                           // 00000000189C: 100604FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018A4: BF870112
	v_add_nc_u32_e32 v1, 0x33000000, v1                        // 0000000018A8: 4A0202FF 33000000
	v_fma_f32 v4, 0x3fc90fda, v2, -v3                          // 0000000018B0: D6130004 840E04FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018BC: BF870112
	v_or_b32_e32 v1, s6, v1                                    // 0000000018C0: 38020206
	v_fmac_f32_e32 v4, 0x33a22168, v2                          // 0000000018C4: 560804FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018CC: BF870091
	v_fmac_f32_e32 v4, 0x3fc90fda, v1                          // 0000000018D0: 560802FF 3FC90FDA
	v_add_f32_e32 v1, v3, v4                                   // 0000000018D8: 06020903
	s_cbranch_execz 2                                          // 0000000018DC: BFA50002 <E_3n6+0x2e8>
	v_mov_b32_e32 v2, s3                                       // 0000000018E0: 7E040203
	s_branch 14                                                // 0000000018E4: BFA0000E <E_3n6+0x320>
	v_mul_f32_e64 v1, 0x3f22f983, s2                           // 0000000018E8: D5080001 000004FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F4: BF870091
	v_rndne_f32_e32 v2, v1                                     // 0000000018F8: 7E044701
	v_fma_f32 v1, 0xbfc90fda, v2, s2                           // 0000000018FC: D6130001 000A04FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001908: BF870091
	v_fmac_f32_e32 v1, 0xb3a22168, v2                          // 00000000190C: 560204FF B3A22168
	v_fmac_f32_e32 v1, 0xa7c234c4, v2                          // 000000001914: 560204FF A7C234C4
	v_cvt_i32_f32_e32 v2, v2                                   // 00000000191C: 7E041102
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001920: BF870141
	v_dual_mul_f32 v3, v1, v1 :: v_dual_lshlrev_b32 v6, 30, v2 // 000000001924: C8E20301 0306049E
	s_mov_b32 s3, 0xb94c1982                                   // 00000000192C: BE8300FF B94C1982
	s_mov_b32 s6, 0x37d75334                                   // 000000001934: BE8600FF 37D75334
	v_xor_b32_e32 v0, s2, v0                                   // 00000000193C: 3A000002
	v_fmaak_f32 v4, s3, v3, 0x3c0881c4                         // 000000001940: 5A080603 3C0881C4
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001948: 84848204
	v_and_b32_e32 v2, 1, v2                                    // 00000000194C: 36040481
	s_add_u32 s0, s0, s4                                       // 000000001950: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001954: 82010501
	v_fmaak_f32 v4, v3, v4, 0xbe2aaa9d                         // 000000001958: 5A080903 BE2AAA9D
	v_fmaak_f32 v5, s6, v3, 0xbab64f3b                         // 000000001960: 5A0A0606 BAB64F3B
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 000000001968: 7C940480
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000196C: BF870193
	v_mul_f32_e32 v4, v3, v4                                   // 000000001970: 10080903
	v_fmaak_f32 v5, v3, v5, 0x3d2aabf7                         // 000000001974: 5A0A0B03 3D2AABF7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000197C: BF870112
	v_dual_fmac_f32 v1, v1, v4 :: v_dual_and_b32 v6, 0x80000000, v6// 000000001980: C8240901 01060CFF 80000000
	v_fmaak_f32 v5, v3, v5, 0xbf000004                         // 00000000198C: 5A0A0B03 BF000004
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001994: BF870112
	v_xor_b32_e32 v0, v0, v6                                   // 000000001998: 3A000D00
	v_fma_f32 v3, v3, v5, 1.0                                  // 00000000199C: D6130003 03CA0B03
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000019A4: BF870121
	v_cndmask_b32_e32 v1, v3, v1, vcc_lo                       // 0000000019A8: 02020303
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 0000000019AC: D47E006A 0001FE02 000001F8
	v_xor_b32_e32 v0, v0, v1                                   // 0000000019B8: 3A000300
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019BC: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_cndmask_b32 v0, 0x7fc00000, v0// 0000000019C0: CA120080 010000FF 7FC00000
	global_store_b32 v1, v0, s[0:1]                            // 0000000019CC: DC6A0000 00000001
	s_nop 0                                                    // 0000000019D4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000019D8: BFB60003
	s_endpgm                                                   // 0000000019DC: BFB00000
