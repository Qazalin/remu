
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n58>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[6:7], s[4:5], 1                               // 000000001614: 84868104
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	global_load_u16 v0, v0, s[2:3]                             // 000000001624: DC4A0000 00020000
	s_waitcnt vmcnt(0)                                         // 00000000162C: BF8903F7
	v_cvt_f32_u32_e32 v0, v0                                   // 000000001630: 7E000D00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001634: BF870091
	v_sub_f32_e32 v1, 0x3fc90fdb, v0                           // 000000001638: 080200FF 3FC90FDB
	v_cmp_ngt_f32_e64 s2, 0x48000000, |v1|                     // 000000001640: D41B0202 000202FF 48000000
	v_readfirstlane_b32 s3, v1                                 // 00000000164C: 7E060501
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001650: BF870092
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001654: 8B6A027E
	s_and_b32 s2, s3, 0x7fffffff                               // 000000001658: 8B02FF03 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001660: BFA300A4 <E_3n58+0x2f4>
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
	v_mov_b32_e32 v2, s6                                       // 00000000179C: 7E040206
	s_cselect_b32 s8, -1, 0                                    // 0000000017A0: 980880C1
	v_alignbit_b32 v3, s3, s11, v2                             // 0000000017A4: D6160003 04081603
	v_alignbit_b32 v4, s11, s9, v2                             // 0000000017AC: D6160004 0408120B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017B4: BF870112
	v_readfirstlane_b32 s6, v3                                 // 0000000017B8: 7E0C0503
	v_cndmask_b32_e64 v3, v4, s11, s8                          // 0000000017BC: D5010003 00201704
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017C4: BF870002
	s_cselect_b32 s3, s3, s6                                   // 0000000017C8: 98030603
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017CC: BF870481
	v_alignbit_b32 v4, s3, v3, 30                              // 0000000017D0: D6160004 027A0603
	s_bfe_u32 s10, s3, 0x1001d                                 // 0000000017D8: 930AFF03 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017E0: BF870009
	s_sub_i32 s6, 0, s10                                       // 0000000017E4: 81860A80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017E8: BF870481
	v_xor_b32_e32 v4, s6, v4                                   // 0000000017EC: 3A080806
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F0: BF870091
	v_clz_i32_u32_e32 v5, v4                                   // 0000000017F4: 7E0A7304
	v_min_u32_e32 v5, 32, v5                                   // 0000000017F8: 260A0AA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000017FC: BF870131
	v_lshlrev_b32_e32 v7, 23, v5                               // 000000001800: 300E0A97
	v_alignbit_b32 v2, s9, s7, v2                              // 000000001804: D6160002 04080E09
	v_sub_nc_u32_e32 v6, 31, v5                                // 00000000180C: 4C0C0A9F
	v_cndmask_b32_e64 v2, v2, s9, s8                           // 000000001810: D5010002 00201302
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001818: BF870121
	v_alignbit_b32 v3, v3, v2, 30                              // 00000000181C: D6160003 027A0503
	v_alignbit_b32 v2, v2, s7, 30                              // 000000001824: D6160002 02780F02
	v_xor_b32_e32 v3, s6, v3                                   // 00000000182C: 3A060606
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001830: BF870002
	v_xor_b32_e32 v2, s6, v2                                   // 000000001834: 3A040406
	s_lshr_b32 s6, s3, 29                                      // 000000001838: 85069D03
	s_lshr_b32 s3, s3, 30                                      // 00000000183C: 85039E03
	s_lshl_b32 s6, s6, 31                                      // 000000001840: 84069F06
	v_alignbit_b32 v4, v4, v3, v6                              // 000000001844: D6160004 041A0704
	v_alignbit_b32 v2, v3, v2, v6                              // 00000000184C: D6160002 041A0503
	s_or_b32 s7, s6, 0.5                                       // 000000001854: 8C07F006
	s_add_i32 s3, s10, s3                                      // 000000001858: 8103030A
	v_sub_nc_u32_e32 v7, s7, v7                                // 00000000185C: 4C0E0E07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001860: BF870092
	v_alignbit_b32 v3, v4, v2, 9                               // 000000001864: D6160003 02260504
	v_clz_i32_u32_e32 v6, v3                                   // 00000000186C: 7E0C7303
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001870: BF870091
	v_min_u32_e32 v6, 32, v6                                   // 000000001874: 260C0CA0
	v_sub_nc_u32_e32 v8, 31, v6                                // 000000001878: 4C100C9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000187C: BF870121
	v_alignbit_b32 v2, v3, v2, v8                              // 000000001880: D6160002 04220503
	v_lshrrev_b32_e32 v3, 9, v4                                // 000000001888: 32060889
	v_lshrrev_b32_e32 v2, 9, v2                                // 00000000188C: 32040489
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001890: BF8700A2
	v_or_b32_e32 v3, v3, v7                                    // 000000001894: 38060F03
	v_add_nc_u32_e32 v5, v6, v5                                // 000000001898: 4A0A0B06
	v_lshlrev_b32_e32 v4, 23, v5                               // 00000000189C: 30080A97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000018A0: BF870211
	v_sub_nc_u32_e32 v2, v2, v4                                // 0000000018A4: 4C040902
	v_mul_f32_e32 v4, 0x3fc90fda, v3                           // 0000000018A8: 100806FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018B0: BF870112
	v_add_nc_u32_e32 v2, 0x33000000, v2                        // 0000000018B4: 4A0404FF 33000000
	v_fma_f32 v5, 0x3fc90fda, v3, -v4                          // 0000000018BC: D6130005 841206FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018C8: BF870112
	v_or_b32_e32 v2, s6, v2                                    // 0000000018CC: 38040406
	v_fmac_f32_e32 v5, 0x33a22168, v3                          // 0000000018D0: 560A06FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D8: BF870091
	v_fmac_f32_e32 v5, 0x3fc90fda, v2                          // 0000000018DC: 560A04FF 3FC90FDA
	v_add_f32_e32 v2, v4, v5                                   // 0000000018E4: 06040B04
	s_cbranch_execz 2                                          // 0000000018E8: BFA50002 <E_3n58+0x2f4>
	v_mov_b32_e32 v3, s3                                       // 0000000018EC: 7E060203
	s_branch 14                                                // 0000000018F0: BFA0000E <E_3n58+0x32c>
	v_mul_f32_e64 v2, 0x3f22f983, s2                           // 0000000018F4: D5080002 000004FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001900: BF870091
	v_rndne_f32_e32 v3, v2                                     // 000000001904: 7E064702
	v_fma_f32 v2, 0xbfc90fda, v3, s2                           // 000000001908: D6130002 000A06FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001914: BF870091
	v_fmac_f32_e32 v2, 0xb3a22168, v3                          // 000000001918: 560406FF B3A22168
	v_fmac_f32_e32 v2, 0xa7c234c4, v3                          // 000000001920: 560406FF A7C234C4
	v_cvt_i32_f32_e32 v3, v3                                   // 000000001928: 7E061103
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000192C: BF870141
	v_dual_mul_f32 v4, 0x3f22f983, v0 :: v_dual_and_b32 v7, 1, v3// 000000001930: C8E400FF 04060681 3F22F983
	s_mov_b32 s3, 0xb94c1982                                   // 00000000193C: BE8300FF B94C1982
	s_mov_b32 s6, 0x37d75334                                   // 000000001944: BE8600FF 37D75334
	v_xor_b32_e32 v1, s2, v1                                   // 00000000194C: 3A020202
	v_rndne_f32_e32 v4, v4                                     // 000000001950: 7E084704
	v_cmp_eq_u32_e32 vcc_lo, 0, v7                             // 000000001954: 7C940E80
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001958: BF870092
	v_dual_mul_f32 v6, v2, v2 :: v_dual_fmamk_f32 v5, v4, 0xbfc90fda, v0// 00000000195C: C8C40502 06040104 BFC90FDA
	v_fmaak_f32 v9, s3, v6, 0x3c0881c4                         // 000000001968: 5A120C03 3C0881C4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001970: BF870091
	v_fmaak_f32 v9, v6, v9, 0xbe2aaa9d                         // 000000001974: 5A121306 BE2AAA9D
	v_dual_fmaak_f32 v10, s6, v6, 0xbab64f3b :: v_dual_mul_f32 v9, v6, v9// 00000000197C: C8460C06 0A081306 BAB64F3B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001988: BF870111
	v_fmaak_f32 v10, v6, v10, 0x3d2aabf7                       // 00000000198C: 5A141506 3D2AABF7
	v_dual_fmac_f32 v5, 0xb3a22168, v4 :: v_dual_fmac_f32 v2, v2, v9// 000000001994: C80008FF 05021302 B3A22168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019A0: BF870092
	v_fmaak_f32 v10, v6, v10, 0xbf000004                       // 0000000019A4: 5A141506 BF000004
	v_fma_f32 v6, v6, v10, 1.0                                 // 0000000019AC: D6130006 03CA1506
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000019B4: BF8701A3
	v_fmac_f32_e32 v5, 0xa7c234c4, v4                          // 0000000019B8: 560A08FF A7C234C4
	v_cvt_i32_f32_e32 v4, v4                                   // 0000000019C0: 7E081104
	v_dual_cndmask_b32 v2, v6, v2 :: v_dual_lshlrev_b32 v3, 30, v3// 0000000019C4: CA620506 0202069E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019CC: BF870091
	v_dual_mul_f32 v8, v5, v5 :: v_dual_and_b32 v3, 0x80000000, v3// 0000000019D0: C8E40B05 080206FF 80000000
	v_fmaak_f32 v11, s3, v8, 0x3c0881c4                        // 0000000019DC: 5A161003 3C0881C4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000019E4: BF870112
	v_xor_b32_e32 v1, v1, v3                                   // 0000000019E8: 3A020701
	v_fmaak_f32 v3, v8, v11, 0xbe2aaa9d                        // 0000000019EC: 5A061708 BE2AAA9D
	v_fmaak_f32 v12, s6, v8, 0xbab64f3b                        // 0000000019F4: 5A181006 BAB64F3B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000019FC: BF870193
	v_xor_b32_e32 v1, v1, v2                                   // 000000001A00: 3A020501
	v_mul_f32_e32 v3, v8, v3                                   // 000000001A04: 10060708
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001A08: BF8701A3
	v_fmaak_f32 v11, v8, v12, 0x3d2aabf7                       // 000000001A0C: 5A161908 3D2AABF7
	v_lshlrev_b32_e32 v12, 30, v4                              // 000000001A14: 3018089E
	v_dual_fmac_f32 v5, v5, v3 :: v_dual_and_b32 v4, 1, v4     // 000000001A18: C8240705 05040881
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001A20: BF870113
	v_fmaak_f32 v11, v8, v11, 0xbf000004                       // 000000001A24: 5A161708 BF000004
	v_cmp_eq_u32_e32 vcc_lo, 0, v4                             // 000000001A2C: 7C940880
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A30: BF870092
	v_fma_f32 v3, v8, v11, 1.0                                 // 000000001A34: D6130003 03CA1708
	v_dual_cndmask_b32 v3, v3, v5 :: v_dual_and_b32 v8, 0x80000000, v12// 000000001A3C: CA640B03 030818FF 80000000
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 000000001A48: D47E006A 0001FE02 000001F8
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001A54: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001A58: BF870119
	s_add_u32 s0, s0, s2                                       // 000000001A5C: 80000200
	v_xor_b32_e32 v2, v8, v3                                   // 000000001A60: 3A040708
	v_cndmask_b32_e32 v1, 0x7fc00000, v1, vcc_lo               // 000000001A64: 020202FF 7FC00000
	v_cmp_class_f32_e64 vcc_lo, v0, 0x1f8                      // 000000001A6C: D47E006A 0001FF00 000001F8
	s_addc_u32 s1, s1, s3                                      // 000000001A78: 82010301
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A7C: BF870093
	v_cndmask_b32_e32 v0, 0x7fc00000, v2, vcc_lo               // 000000001A80: 020004FF 7FC00000
	v_div_scale_f32 v2, null, v1, v1, v0                       // 000000001A88: D6FC7C02 04020301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001A90: BF8700B1
	v_rcp_f32_e32 v3, v2                                       // 000000001A94: 7E065502
	s_waitcnt_depctr 0xfff                                     // 000000001A98: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 000000001A9C: D6130004 23CA0702
	v_fmac_f32_e32 v3, v4, v3                                  // 000000001AA4: 56060704
	v_div_scale_f32 v5, vcc_lo, v0, v1, v0                     // 000000001AA8: D6FC6A05 04020300
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AB0: BF870091
	v_mul_f32_e32 v4, v5, v3                                   // 000000001AB4: 10080705
	v_fma_f32 v6, -v2, v4, v5                                  // 000000001AB8: D6130006 24160902
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AC0: BF870091
	v_fmac_f32_e32 v4, v6, v3                                  // 000000001AC4: 56080706
	v_fma_f32 v2, -v2, v4, v5                                  // 000000001AC8: D6130002 24160902
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AD0: BF870091
	v_div_fmas_f32 v2, v2, v3, v4                              // 000000001AD4: D6370002 04120702
	v_div_fixup_f32 v0, v2, v1, v0                             // 000000001ADC: D6270000 04020302
	v_mov_b32_e32 v1, 0                                        // 000000001AE4: 7E020280
	global_store_b32 v1, v0, s[0:1]                            // 000000001AE8: DC6A0000 00000001
	s_nop 0                                                    // 000000001AF0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001AF4: BFB60003
	s_endpgm                                                   // 000000001AF8: BFB00000
