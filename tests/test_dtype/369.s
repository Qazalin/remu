
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n17>:
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
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001634: BF870091
	v_sub_f32_e32 v0, 0x3fc90fdb, v0                           // 000000001638: 080000FF 3FC90FDB
	v_cmp_ngt_f32_e64 s2, 0x48000000, |v0|                     // 000000001640: D41B0202 000200FF 48000000
	v_readfirstlane_b32 s3, v0                                 // 00000000164C: 7E060500
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001650: BF870092
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001654: 8B6A027E
	s_and_b32 s2, s3, 0x7fffffff                               // 000000001658: 8B02FF03 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001660: BFA300A4 <E_3n17+0x2f4>
	s_and_b32 s3, s2, 0x7fffff                                 // 000000001664: 8B03FF02 007FFFFF
	s_lshr_b32 s6, s2, 23                                      // 00000000166C: 85069702
	s_bitset1_b32 s3, 23                                       // 000000001670: BE831297
	s_addk_i32 s6, 0xff88                                      // 000000001674: B786FF88
	s_mul_hi_u32 s7, s3, 0xfe5163ab                            // 000000001678: 9687FF03 FE5163AB
	s_mul_i32 s8, s3, 0x3c439041                               // 000000001680: 9608FF03 3C439041
	s_mul_hi_u32 s9, s3, 0x3c439041                            // 000000001688: 9689FF03 3C439041
	s_add_u32 s7, s7, s8                                       // 000000001690: 80070807
	s_addc_u32 s8, 0, s9                                       // 000000001694: 82080980
	s_mul_i32 s9, s3, 0xdb629599                               // 000000001698: 9609FF03 DB629599
	s_mul_hi_u32 s10, s3, 0xdb629599                           // 0000000016A0: 968AFF03 DB629599
	s_add_u32 s8, s8, s9                                       // 0000000016A8: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000016AC: 82090A80
	s_mul_i32 s10, s3, 0xf534ddc0                              // 0000000016B0: 960AFF03 F534DDC0
	s_mul_hi_u32 s11, s3, 0xf534ddc0                           // 0000000016B8: 968BFF03 F534DDC0
	s_add_u32 s9, s9, s10                                      // 0000000016C0: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000016C4: 820A0B80
	s_mul_i32 s11, s3, 0xfc2757d1                              // 0000000016C8: 960BFF03 FC2757D1
	s_mul_hi_u32 s12, s3, 0xfc2757d1                           // 0000000016D0: 968CFF03 FC2757D1
	s_add_u32 s10, s10, s11                                    // 0000000016D8: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000016DC: 820B0C80
	s_mul_i32 s12, s3, 0x4e441529                              // 0000000016E0: 960CFF03 4E441529
	s_mul_hi_u32 s13, s3, 0x4e441529                           // 0000000016E8: 968DFF03 4E441529
	s_add_u32 s11, s11, s12                                    // 0000000016F0: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000016F4: 820C0D80
	s_cmp_gt_u32 s6, 63                                        // 0000000016F8: BF08BF06
	s_mul_i32 s13, s3, 0xfe5163ab                              // 0000000016FC: 960DFF03 FE5163AB
	s_mul_hi_u32 s14, s3, 0xa2f9836e                           // 000000001704: 968EFF03 A2F9836E
	s_mul_i32 s3, s3, 0xa2f9836e                               // 00000000170C: 9603FF03 A2F9836E
	s_cselect_b32 s15, s8, s10                                 // 000000001714: 980F0A08
	s_cselect_b32 s7, s7, s9                                   // 000000001718: 98070907
	s_cselect_b32 s8, s13, s8                                  // 00000000171C: 9808080D
	s_add_u32 s3, s12, s3                                      // 000000001720: 8003030C
	s_addc_u32 s12, 0, s14                                     // 000000001724: 820C0E80
	s_cmp_gt_u32 s6, 63                                        // 000000001728: BF08BF06
	s_cselect_b32 s13, 0xffffffc0, 0                           // 00000000172C: 980D80FF FFFFFFC0
	s_cselect_b32 s9, s9, s11                                  // 000000001734: 98090B09
	s_cselect_b32 s3, s10, s3                                  // 000000001738: 9803030A
	s_cselect_b32 s10, s11, s12                                // 00000000173C: 980A0C0B
	s_add_i32 s13, s13, s6                                     // 000000001740: 810D060D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001744: BF870009
	s_cmp_gt_u32 s13, 31                                       // 000000001748: BF089F0D
	s_cselect_b32 s6, 0xffffffe0, 0                            // 00000000174C: 980680FF FFFFFFE0
	s_cselect_b32 s11, s9, s3                                  // 000000001754: 980B0309
	s_cselect_b32 s3, s3, s10                                  // 000000001758: 98030A03
	s_cselect_b32 s9, s15, s9                                  // 00000000175C: 9809090F
	s_cselect_b32 s10, s7, s15                                 // 000000001760: 980A0F07
	s_cselect_b32 s7, s8, s7                                   // 000000001764: 98070708
	s_add_i32 s6, s6, s13                                      // 000000001768: 81060D06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000176C: BF870009
	s_cmp_gt_u32 s6, 31                                        // 000000001770: BF089F06
	s_cselect_b32 s8, 0xffffffe0, 0                            // 000000001774: 980880FF FFFFFFE0
	s_cselect_b32 s3, s11, s3                                  // 00000000177C: 9803030B
	s_cselect_b32 s11, s9, s11                                 // 000000001780: 980B0B09
	s_cselect_b32 s9, s10, s9                                  // 000000001784: 9809090A
	s_cselect_b32 s7, s7, s10                                  // 000000001788: 98070A07
	s_add_i32 s8, s8, s6                                       // 00000000178C: 81080608
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001790: BF8700C9
	s_sub_i32 s6, 32, s8                                       // 000000001794: 818608A0
	s_cmp_eq_u32 s8, 0                                         // 000000001798: BF068008
	v_mov_b32_e32 v1, s6                                       // 00000000179C: 7E020206
	s_cselect_b32 s8, -1, 0                                    // 0000000017A0: 980880C1
	v_alignbit_b32 v2, s3, s11, v1                             // 0000000017A4: D6160002 04041603
	v_alignbit_b32 v3, s11, s9, v1                             // 0000000017AC: D6160003 0404120B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017B4: BF870112
	v_readfirstlane_b32 s6, v2                                 // 0000000017B8: 7E0C0502
	v_cndmask_b32_e64 v2, v3, s11, s8                          // 0000000017BC: D5010002 00201703
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017C4: BF870002
	s_cselect_b32 s3, s3, s6                                   // 0000000017C8: 98030603
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017CC: BF870481
	v_alignbit_b32 v3, s3, v2, 30                              // 0000000017D0: D6160003 027A0403
	s_bfe_u32 s10, s3, 0x1001d                                 // 0000000017D8: 930AFF03 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017E0: BF870009
	s_sub_i32 s6, 0, s10                                       // 0000000017E4: 81860A80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017E8: BF870481
	v_xor_b32_e32 v3, s6, v3                                   // 0000000017EC: 3A060606
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F0: BF870091
	v_clz_i32_u32_e32 v4, v3                                   // 0000000017F4: 7E087303
	v_min_u32_e32 v4, 32, v4                                   // 0000000017F8: 260808A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000017FC: BF870131
	v_lshlrev_b32_e32 v6, 23, v4                               // 000000001800: 300C0897
	v_alignbit_b32 v1, s9, s7, v1                              // 000000001804: D6160001 04040E09
	v_sub_nc_u32_e32 v5, 31, v4                                // 00000000180C: 4C0A089F
	v_cndmask_b32_e64 v1, v1, s9, s8                           // 000000001810: D5010001 00201301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001818: BF870121
	v_alignbit_b32 v2, v2, v1, 30                              // 00000000181C: D6160002 027A0302
	v_alignbit_b32 v1, v1, s7, 30                              // 000000001824: D6160001 02780F01
	v_xor_b32_e32 v2, s6, v2                                   // 00000000182C: 3A040406
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001830: BF870002
	v_xor_b32_e32 v1, s6, v1                                   // 000000001834: 3A020206
	s_lshr_b32 s6, s3, 29                                      // 000000001838: 85069D03
	s_lshr_b32 s3, s3, 30                                      // 00000000183C: 85039E03
	s_lshl_b32 s6, s6, 31                                      // 000000001840: 84069F06
	v_alignbit_b32 v3, v3, v2, v5                              // 000000001844: D6160003 04160503
	v_alignbit_b32 v1, v2, v1, v5                              // 00000000184C: D6160001 04160302
	s_or_b32 s7, s6, 0.5                                       // 000000001854: 8C07F006
	s_add_i32 s3, s10, s3                                      // 000000001858: 8103030A
	v_sub_nc_u32_e32 v6, s7, v6                                // 00000000185C: 4C0C0C07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001860: BF870092
	v_alignbit_b32 v2, v3, v1, 9                               // 000000001864: D6160002 02260303
	v_clz_i32_u32_e32 v5, v2                                   // 00000000186C: 7E0A7302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001870: BF870091
	v_min_u32_e32 v5, 32, v5                                   // 000000001874: 260A0AA0
	v_sub_nc_u32_e32 v7, 31, v5                                // 000000001878: 4C0E0A9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000187C: BF870121
	v_alignbit_b32 v1, v2, v1, v7                              // 000000001880: D6160001 041E0302
	v_lshrrev_b32_e32 v2, 9, v3                                // 000000001888: 32040689
	v_lshrrev_b32_e32 v1, 9, v1                                // 00000000188C: 32020289
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001890: BF8700A2
	v_or_b32_e32 v2, v2, v6                                    // 000000001894: 38040D02
	v_add_nc_u32_e32 v4, v5, v4                                // 000000001898: 4A080905
	v_lshlrev_b32_e32 v3, 23, v4                               // 00000000189C: 30060897
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000018A0: BF870211
	v_sub_nc_u32_e32 v1, v1, v3                                // 0000000018A4: 4C020701
	v_mul_f32_e32 v3, 0x3fc90fda, v2                           // 0000000018A8: 100604FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018B0: BF870112
	v_add_nc_u32_e32 v1, 0x33000000, v1                        // 0000000018B4: 4A0202FF 33000000
	v_fma_f32 v4, 0x3fc90fda, v2, -v3                          // 0000000018BC: D6130004 840E04FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018C8: BF870112
	v_or_b32_e32 v1, s6, v1                                    // 0000000018CC: 38020206
	v_fmac_f32_e32 v4, 0x33a22168, v2                          // 0000000018D0: 560804FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D8: BF870091
	v_fmac_f32_e32 v4, 0x3fc90fda, v1                          // 0000000018DC: 560802FF 3FC90FDA
	v_add_f32_e32 v1, v3, v4                                   // 0000000018E4: 06020903
	s_cbranch_execz 2                                          // 0000000018E8: BFA50002 <E_3n17+0x2f4>
	v_mov_b32_e32 v2, s3                                       // 0000000018EC: 7E040203
	s_branch 14                                                // 0000000018F0: BFA0000E <E_3n17+0x32c>
	v_mul_f32_e64 v1, 0x3f22f983, s2                           // 0000000018F4: D5080001 000004FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001900: BF870091
	v_rndne_f32_e32 v2, v1                                     // 000000001904: 7E044701
	v_fma_f32 v1, 0xbfc90fda, v2, s2                           // 000000001908: D6130001 000A04FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001914: BF870091
	v_fmac_f32_e32 v1, 0xb3a22168, v2                          // 000000001918: 560204FF B3A22168
	v_fmac_f32_e32 v1, 0xa7c234c4, v2                          // 000000001920: 560204FF A7C234C4
	v_cvt_i32_f32_e32 v2, v2                                   // 000000001928: 7E041102
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000192C: BF870141
	v_dual_mul_f32 v3, v1, v1 :: v_dual_lshlrev_b32 v6, 30, v2 // 000000001930: C8E20301 0306049E
	s_mov_b32 s3, 0xb94c1982                                   // 000000001938: BE8300FF B94C1982
	s_mov_b32 s6, 0x37d75334                                   // 000000001940: BE8600FF 37D75334
	v_xor_b32_e32 v0, s2, v0                                   // 000000001948: 3A000002
	v_fmaak_f32 v4, s3, v3, 0x3c0881c4                         // 00000000194C: 5A080603 3C0881C4
	s_add_u32 s0, s0, s4                                       // 000000001954: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001958: 82010501
	v_and_b32_e32 v2, 1, v2                                    // 00000000195C: 36040481
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001960: BF8701A2
	v_fmaak_f32 v4, v3, v4, 0xbe2aaa9d                         // 000000001964: 5A080903 BE2AAA9D
	v_fmaak_f32 v5, s6, v3, 0xbab64f3b                         // 00000000196C: 5A0A0606 BAB64F3B
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 000000001974: 7C940480
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001978: BF870193
	v_mul_f32_e32 v4, v3, v4                                   // 00000000197C: 10080903
	v_fmaak_f32 v5, v3, v5, 0x3d2aabf7                         // 000000001980: 5A0A0B03 3D2AABF7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001988: BF870112
	v_dual_fmac_f32 v1, v1, v4 :: v_dual_and_b32 v6, 0x80000000, v6// 00000000198C: C8240901 01060CFF 80000000
	v_fmaak_f32 v5, v3, v5, 0xbf000004                         // 000000001998: 5A0A0B03 BF000004
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000019A0: BF870112
	v_xor_b32_e32 v0, v0, v6                                   // 0000000019A4: 3A000D00
	v_fma_f32 v3, v3, v5, 1.0                                  // 0000000019A8: D6130003 03CA0B03
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000019B0: BF870121
	v_cndmask_b32_e32 v1, v3, v1, vcc_lo                       // 0000000019B4: 02020303
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 0000000019B8: D47E006A 0001FE02 000001F8
	v_xor_b32_e32 v0, v0, v1                                   // 0000000019C4: 3A000300
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019C8: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_cndmask_b32 v0, 0x7fc00000, v0// 0000000019CC: CA120080 010000FF 7FC00000
	global_store_b32 v1, v0, s[0:1]                            // 0000000019D8: DC6A0000 00000001
	s_nop 0                                                    // 0000000019E0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000019E4: BFB60003
	s_endpgm                                                   // 0000000019E8: BFB00000
