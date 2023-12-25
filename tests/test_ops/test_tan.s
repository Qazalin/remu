
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n45>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_load_b32 s3, s[2:3], null                                // 000000001624: F40000C1 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	v_cmp_ngt_f32_e64 s2, 0x48000000, |s3|                     // 000000001630: D41B0202 000006FF 48000000
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000163C: BF870001
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001640: 8B6A027E
	s_and_b32 s2, s3, 0x7fffffff                               // 000000001644: 8B02FF03 7FFFFFFF
	s_cbranch_vccz 164                                         // 00000000164C: BFA300A4 <E_2925n45+0x2e0>
	s_and_b32 s6, s2, 0x7fffff                                 // 000000001650: 8B06FF02 007FFFFF
	s_lshr_b32 s7, s2, 23                                      // 000000001658: 85079702
	s_bitset1_b32 s6, 23                                       // 00000000165C: BE861297
	s_addk_i32 s7, 0xff88                                      // 000000001660: B787FF88
	s_mul_hi_u32 s8, s6, 0xfe5163ab                            // 000000001664: 9688FF06 FE5163AB
	s_mul_i32 s9, s6, 0x3c439041                               // 00000000166C: 9609FF06 3C439041
	s_mul_hi_u32 s10, s6, 0x3c439041                           // 000000001674: 968AFF06 3C439041
	s_add_u32 s8, s8, s9                                       // 00000000167C: 80080908
	s_addc_u32 s9, 0, s10                                      // 000000001680: 82090A80
	s_mul_i32 s10, s6, 0xdb629599                              // 000000001684: 960AFF06 DB629599
	s_mul_hi_u32 s11, s6, 0xdb629599                           // 00000000168C: 968BFF06 DB629599
	s_add_u32 s9, s9, s10                                      // 000000001694: 80090A09
	s_addc_u32 s10, 0, s11                                     // 000000001698: 820A0B80
	s_mul_i32 s11, s6, 0xf534ddc0                              // 00000000169C: 960BFF06 F534DDC0
	s_mul_hi_u32 s12, s6, 0xf534ddc0                           // 0000000016A4: 968CFF06 F534DDC0
	s_add_u32 s10, s10, s11                                    // 0000000016AC: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000016B0: 820B0C80
	s_mul_i32 s12, s6, 0xfc2757d1                              // 0000000016B4: 960CFF06 FC2757D1
	s_mul_hi_u32 s13, s6, 0xfc2757d1                           // 0000000016BC: 968DFF06 FC2757D1
	s_add_u32 s11, s11, s12                                    // 0000000016C4: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000016C8: 820C0D80
	s_mul_i32 s13, s6, 0x4e441529                              // 0000000016CC: 960DFF06 4E441529
	s_mul_hi_u32 s14, s6, 0x4e441529                           // 0000000016D4: 968EFF06 4E441529
	s_add_u32 s12, s12, s13                                    // 0000000016DC: 800C0D0C
	s_addc_u32 s13, 0, s14                                     // 0000000016E0: 820D0E80
	s_cmp_gt_u32 s7, 63                                        // 0000000016E4: BF08BF07
	s_mul_i32 s14, s6, 0xfe5163ab                              // 0000000016E8: 960EFF06 FE5163AB
	s_mul_hi_u32 s15, s6, 0xa2f9836e                           // 0000000016F0: 968FFF06 A2F9836E
	s_mul_i32 s6, s6, 0xa2f9836e                               // 0000000016F8: 9606FF06 A2F9836E
	s_cselect_b32 s16, s9, s11                                 // 000000001700: 98100B09
	s_cselect_b32 s8, s8, s10                                  // 000000001704: 98080A08
	s_cselect_b32 s9, s14, s9                                  // 000000001708: 9809090E
	s_add_u32 s6, s13, s6                                      // 00000000170C: 8006060D
	s_addc_u32 s13, 0, s15                                     // 000000001710: 820D0F80
	s_cmp_gt_u32 s7, 63                                        // 000000001714: BF08BF07
	s_cselect_b32 s14, 0xffffffc0, 0                           // 000000001718: 980E80FF FFFFFFC0
	s_cselect_b32 s10, s10, s12                                // 000000001720: 980A0C0A
	s_cselect_b32 s6, s11, s6                                  // 000000001724: 9806060B
	s_cselect_b32 s11, s12, s13                                // 000000001728: 980B0D0C
	s_add_i32 s14, s14, s7                                     // 00000000172C: 810E070E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001730: BF870009
	s_cmp_gt_u32 s14, 31                                       // 000000001734: BF089F0E
	s_cselect_b32 s7, 0xffffffe0, 0                            // 000000001738: 980780FF FFFFFFE0
	s_cselect_b32 s12, s10, s6                                 // 000000001740: 980C060A
	s_cselect_b32 s6, s6, s11                                  // 000000001744: 98060B06
	s_cselect_b32 s10, s16, s10                                // 000000001748: 980A0A10
	s_cselect_b32 s11, s8, s16                                 // 00000000174C: 980B1008
	s_cselect_b32 s8, s9, s8                                   // 000000001750: 98080809
	s_add_i32 s7, s7, s14                                      // 000000001754: 81070E07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001758: BF870009
	s_cmp_gt_u32 s7, 31                                        // 00000000175C: BF089F07
	s_cselect_b32 s9, 0xffffffe0, 0                            // 000000001760: 980980FF FFFFFFE0
	s_cselect_b32 s6, s12, s6                                  // 000000001768: 9806060C
	s_cselect_b32 s12, s10, s12                                // 00000000176C: 980C0C0A
	s_cselect_b32 s10, s11, s10                                // 000000001770: 980A0A0B
	s_cselect_b32 s8, s8, s11                                  // 000000001774: 98080B08
	s_add_i32 s9, s9, s7                                       // 000000001778: 81090709
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000177C: BF8700C9
	s_sub_i32 s7, 32, s9                                       // 000000001780: 818709A0
	s_cmp_eq_u32 s9, 0                                         // 000000001784: BF068009
	v_mov_b32_e32 v0, s7                                       // 000000001788: 7E000207
	s_cselect_b32 s9, -1, 0                                    // 00000000178C: 980980C1
	v_alignbit_b32 v1, s6, s12, v0                             // 000000001790: D6160001 04001806
	v_alignbit_b32 v2, s12, s10, v0                            // 000000001798: D6160002 0400140C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017A0: BF870112
	v_readfirstlane_b32 s7, v1                                 // 0000000017A4: 7E0E0501
	v_cndmask_b32_e64 v1, v2, s12, s9                          // 0000000017A8: D5010001 00241902
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017B0: BF870002
	s_cselect_b32 s6, s6, s7                                   // 0000000017B4: 98060706
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017B8: BF870481
	v_alignbit_b32 v2, s6, v1, 30                              // 0000000017BC: D6160002 027A0206
	s_bfe_u32 s11, s6, 0x1001d                                 // 0000000017C4: 930BFF06 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017CC: BF870009
	s_sub_i32 s7, 0, s11                                       // 0000000017D0: 81870B80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017D4: BF870481
	v_xor_b32_e32 v2, s7, v2                                   // 0000000017D8: 3A040407
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017DC: BF870091
	v_clz_i32_u32_e32 v3, v2                                   // 0000000017E0: 7E067302
	v_min_u32_e32 v3, 32, v3                                   // 0000000017E4: 260606A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000017E8: BF870131
	v_lshlrev_b32_e32 v5, 23, v3                               // 0000000017EC: 300A0697
	v_alignbit_b32 v0, s10, s8, v0                             // 0000000017F0: D6160000 0400100A
	v_sub_nc_u32_e32 v4, 31, v3                                // 0000000017F8: 4C08069F
	v_cndmask_b32_e64 v0, v0, s10, s9                          // 0000000017FC: D5010000 00241500
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001804: BF870121
	v_alignbit_b32 v1, v1, v0, 30                              // 000000001808: D6160001 027A0101
	v_alignbit_b32 v0, v0, s8, 30                              // 000000001810: D6160000 02781100
	v_xor_b32_e32 v1, s7, v1                                   // 000000001818: 3A020207
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000181C: BF870002
	v_xor_b32_e32 v0, s7, v0                                   // 000000001820: 3A000007
	s_lshr_b32 s7, s6, 29                                      // 000000001824: 85079D06
	s_lshr_b32 s6, s6, 30                                      // 000000001828: 85069E06
	s_lshl_b32 s7, s7, 31                                      // 00000000182C: 84079F07
	v_alignbit_b32 v2, v2, v1, v4                              // 000000001830: D6160002 04120302
	v_alignbit_b32 v0, v1, v0, v4                              // 000000001838: D6160000 04120101
	s_or_b32 s8, s7, 0.5                                       // 000000001840: 8C08F007
	s_add_i32 s6, s11, s6                                      // 000000001844: 8106060B
	v_sub_nc_u32_e32 v5, s8, v5                                // 000000001848: 4C0A0A08
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000184C: BF870092
	v_alignbit_b32 v1, v2, v0, 9                               // 000000001850: D6160001 02260102
	v_clz_i32_u32_e32 v4, v1                                   // 000000001858: 7E087301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000185C: BF870091
	v_min_u32_e32 v4, 32, v4                                   // 000000001860: 260808A0
	v_sub_nc_u32_e32 v6, 31, v4                                // 000000001864: 4C0C089F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001868: BF870121
	v_alignbit_b32 v0, v1, v0, v6                              // 00000000186C: D6160000 041A0101
	v_lshrrev_b32_e32 v1, 9, v2                                // 000000001874: 32020489
	v_lshrrev_b32_e32 v0, 9, v0                                // 000000001878: 32000089
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000187C: BF8700A2
	v_or_b32_e32 v1, v1, v5                                    // 000000001880: 38020B01
	v_add_nc_u32_e32 v3, v4, v3                                // 000000001884: 4A060704
	v_lshlrev_b32_e32 v2, 23, v3                               // 000000001888: 30040697
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000188C: BF870211
	v_sub_nc_u32_e32 v0, v0, v2                                // 000000001890: 4C000500
	v_mul_f32_e32 v2, 0x3fc90fda, v1                           // 000000001894: 100402FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000189C: BF870112
	v_add_nc_u32_e32 v0, 0x33000000, v0                        // 0000000018A0: 4A0000FF 33000000
	v_fma_f32 v3, 0x3fc90fda, v1, -v2                          // 0000000018A8: D6130003 840A02FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018B4: BF870112
	v_or_b32_e32 v0, s7, v0                                    // 0000000018B8: 38000007
	v_fmac_f32_e32 v3, 0x33a22168, v1                          // 0000000018BC: 560602FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018C4: BF870091
	v_fmac_f32_e32 v3, 0x3fc90fda, v0                          // 0000000018C8: 560600FF 3FC90FDA
	v_add_f32_e32 v0, v2, v3                                   // 0000000018D0: 06000702
	s_cbranch_execz 2                                          // 0000000018D4: BFA50002 <E_2925n45+0x2e0>
	v_mov_b32_e32 v1, s6                                       // 0000000018D8: 7E020206
	s_branch 14                                                // 0000000018DC: BFA0000E <E_2925n45+0x318>
	v_mul_f32_e64 v0, 0x3f22f983, s2                           // 0000000018E0: D5080000 000004FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018EC: BF870091
	v_rndne_f32_e32 v1, v0                                     // 0000000018F0: 7E024700
	v_fma_f32 v0, 0xbfc90fda, v1, s2                           // 0000000018F4: D6130000 000A02FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001900: BF870091
	v_fmac_f32_e32 v0, 0xb3a22168, v1                          // 000000001904: 560002FF B3A22168
	v_fmac_f32_e32 v0, 0xa7c234c4, v1                          // 00000000190C: 560002FF A7C234C4
	v_cvt_i32_f32_e32 v1, v1                                   // 000000001914: 7E021101
	v_sub_f32_e64 v2, 0x3fc90fdb, s3                           // 000000001918: D5040002 000006FF 3FC90FDB
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001924: BF870121
	v_cmp_ngt_f32_e64 s6, 0x48000000, |v2|                     // 000000001928: D41B0206 000204FF 48000000
	v_readfirstlane_b32 s7, v2                                 // 000000001934: 7E0E0502
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001938: 8B6A067E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000193C: BF870001
	s_and_b32 s6, s7, 0x7fffffff                               // 000000001940: 8B06FF07 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001948: BFA300A4 <E_2925n45+0x5dc>
	s_and_b32 s7, s6, 0x7fffff                                 // 00000000194C: 8B07FF06 007FFFFF
	s_lshr_b32 s8, s6, 23                                      // 000000001954: 85089706
	s_bitset1_b32 s7, 23                                       // 000000001958: BE871297
	s_addk_i32 s8, 0xff88                                      // 00000000195C: B788FF88
	s_mul_hi_u32 s9, s7, 0xfe5163ab                            // 000000001960: 9689FF07 FE5163AB
	s_mul_i32 s10, s7, 0x3c439041                              // 000000001968: 960AFF07 3C439041
	s_mul_hi_u32 s11, s7, 0x3c439041                           // 000000001970: 968BFF07 3C439041
	s_add_u32 s9, s9, s10                                      // 000000001978: 80090A09
	s_addc_u32 s10, 0, s11                                     // 00000000197C: 820A0B80
	s_mul_i32 s11, s7, 0xdb629599                              // 000000001980: 960BFF07 DB629599
	s_mul_hi_u32 s12, s7, 0xdb629599                           // 000000001988: 968CFF07 DB629599
	s_add_u32 s10, s10, s11                                    // 000000001990: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 000000001994: 820B0C80
	s_mul_i32 s12, s7, 0xf534ddc0                              // 000000001998: 960CFF07 F534DDC0
	s_mul_hi_u32 s13, s7, 0xf534ddc0                           // 0000000019A0: 968DFF07 F534DDC0
	s_add_u32 s11, s11, s12                                    // 0000000019A8: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000019AC: 820C0D80
	s_mul_i32 s13, s7, 0xfc2757d1                              // 0000000019B0: 960DFF07 FC2757D1
	s_mul_hi_u32 s14, s7, 0xfc2757d1                           // 0000000019B8: 968EFF07 FC2757D1
	s_add_u32 s12, s12, s13                                    // 0000000019C0: 800C0D0C
	s_addc_u32 s13, 0, s14                                     // 0000000019C4: 820D0E80
	s_mul_i32 s14, s7, 0x4e441529                              // 0000000019C8: 960EFF07 4E441529
	s_mul_hi_u32 s15, s7, 0x4e441529                           // 0000000019D0: 968FFF07 4E441529
	s_add_u32 s13, s13, s14                                    // 0000000019D8: 800D0E0D
	s_addc_u32 s14, 0, s15                                     // 0000000019DC: 820E0F80
	s_cmp_gt_u32 s8, 63                                        // 0000000019E0: BF08BF08
	s_mul_i32 s15, s7, 0xfe5163ab                              // 0000000019E4: 960FFF07 FE5163AB
	s_mul_hi_u32 s16, s7, 0xa2f9836e                           // 0000000019EC: 9690FF07 A2F9836E
	s_mul_i32 s7, s7, 0xa2f9836e                               // 0000000019F4: 9607FF07 A2F9836E
	s_cselect_b32 s17, s10, s12                                // 0000000019FC: 98110C0A
	s_cselect_b32 s9, s9, s11                                  // 000000001A00: 98090B09
	s_cselect_b32 s10, s15, s10                                // 000000001A04: 980A0A0F
	s_add_u32 s7, s14, s7                                      // 000000001A08: 8007070E
	s_addc_u32 s14, 0, s16                                     // 000000001A0C: 820E1080
	s_cmp_gt_u32 s8, 63                                        // 000000001A10: BF08BF08
	s_cselect_b32 s15, 0xffffffc0, 0                           // 000000001A14: 980F80FF FFFFFFC0
	s_cselect_b32 s11, s11, s13                                // 000000001A1C: 980B0D0B
	s_cselect_b32 s7, s12, s7                                  // 000000001A20: 9807070C
	s_cselect_b32 s12, s13, s14                                // 000000001A24: 980C0E0D
	s_add_i32 s15, s15, s8                                     // 000000001A28: 810F080F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A2C: BF870009
	s_cmp_gt_u32 s15, 31                                       // 000000001A30: BF089F0F
	s_cselect_b32 s8, 0xffffffe0, 0                            // 000000001A34: 980880FF FFFFFFE0
	s_cselect_b32 s13, s11, s7                                 // 000000001A3C: 980D070B
	s_cselect_b32 s7, s7, s12                                  // 000000001A40: 98070C07
	s_cselect_b32 s11, s17, s11                                // 000000001A44: 980B0B11
	s_cselect_b32 s12, s9, s17                                 // 000000001A48: 980C1109
	s_cselect_b32 s9, s10, s9                                  // 000000001A4C: 9809090A
	s_add_i32 s8, s8, s15                                      // 000000001A50: 81080F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A54: BF870009
	s_cmp_gt_u32 s8, 31                                        // 000000001A58: BF089F08
	s_cselect_b32 s10, 0xffffffe0, 0                           // 000000001A5C: 980A80FF FFFFFFE0
	s_cselect_b32 s7, s13, s7                                  // 000000001A64: 9807070D
	s_cselect_b32 s13, s11, s13                                // 000000001A68: 980D0D0B
	s_cselect_b32 s11, s12, s11                                // 000000001A6C: 980B0B0C
	s_cselect_b32 s9, s9, s12                                  // 000000001A70: 98090C09
	s_add_i32 s10, s10, s8                                     // 000000001A74: 810A080A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001A78: BF8700C9
	s_sub_i32 s8, 32, s10                                      // 000000001A7C: 81880AA0
	s_cmp_eq_u32 s10, 0                                        // 000000001A80: BF06800A
	v_mov_b32_e32 v3, s8                                       // 000000001A84: 7E060208
	s_cselect_b32 s10, -1, 0                                   // 000000001A88: 980A80C1
	v_alignbit_b32 v4, s7, s13, v3                             // 000000001A8C: D6160004 040C1A07
	v_alignbit_b32 v5, s13, s11, v3                            // 000000001A94: D6160005 040C160D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001A9C: BF870112
	v_readfirstlane_b32 s8, v4                                 // 000000001AA0: 7E100504
	v_cndmask_b32_e64 v4, v5, s13, s10                         // 000000001AA4: D5010004 00281B05
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001AAC: BF870002
	s_cselect_b32 s7, s7, s8                                   // 000000001AB0: 98070807
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001AB4: BF870481
	v_alignbit_b32 v5, s7, v4, 30                              // 000000001AB8: D6160005 027A0807
	s_bfe_u32 s12, s7, 0x1001d                                 // 000000001AC0: 930CFF07 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001AC8: BF870009
	s_sub_i32 s8, 0, s12                                       // 000000001ACC: 81880C80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001AD0: BF870481
	v_xor_b32_e32 v5, s8, v5                                   // 000000001AD4: 3A0A0A08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AD8: BF870091
	v_clz_i32_u32_e32 v6, v5                                   // 000000001ADC: 7E0C7305
	v_min_u32_e32 v6, 32, v6                                   // 000000001AE0: 260C0CA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001AE4: BF870131
	v_lshlrev_b32_e32 v8, 23, v6                               // 000000001AE8: 30100C97
	v_alignbit_b32 v3, s11, s9, v3                             // 000000001AEC: D6160003 040C120B
	v_sub_nc_u32_e32 v7, 31, v6                                // 000000001AF4: 4C0E0C9F
	v_cndmask_b32_e64 v3, v3, s11, s10                         // 000000001AF8: D5010003 00281703
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001B00: BF870121
	v_alignbit_b32 v4, v4, v3, 30                              // 000000001B04: D6160004 027A0704
	v_alignbit_b32 v3, v3, s9, 30                              // 000000001B0C: D6160003 02781303
	v_xor_b32_e32 v4, s8, v4                                   // 000000001B14: 3A080808
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B18: BF870002
	v_xor_b32_e32 v3, s8, v3                                   // 000000001B1C: 3A060608
	s_lshr_b32 s8, s7, 29                                      // 000000001B20: 85089D07
	s_lshr_b32 s7, s7, 30                                      // 000000001B24: 85079E07
	s_lshl_b32 s8, s8, 31                                      // 000000001B28: 84089F08
	v_alignbit_b32 v5, v5, v4, v7                              // 000000001B2C: D6160005 041E0905
	v_alignbit_b32 v3, v4, v3, v7                              // 000000001B34: D6160003 041E0704
	s_or_b32 s9, s8, 0.5                                       // 000000001B3C: 8C09F008
	s_add_i32 s7, s12, s7                                      // 000000001B40: 8107070C
	v_sub_nc_u32_e32 v8, s9, v8                                // 000000001B44: 4C101009
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B48: BF870092
	v_alignbit_b32 v4, v5, v3, 9                               // 000000001B4C: D6160004 02260705
	v_clz_i32_u32_e32 v7, v4                                   // 000000001B54: 7E0E7304
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B58: BF870091
	v_min_u32_e32 v7, 32, v7                                   // 000000001B5C: 260E0EA0
	v_sub_nc_u32_e32 v9, 31, v7                                // 000000001B60: 4C120E9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001B64: BF870121
	v_alignbit_b32 v3, v4, v3, v9                              // 000000001B68: D6160003 04260704
	v_lshrrev_b32_e32 v4, 9, v5                                // 000000001B70: 32080A89
	v_lshrrev_b32_e32 v3, 9, v3                                // 000000001B74: 32060689
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B78: BF8700A2
	v_or_b32_e32 v4, v4, v8                                    // 000000001B7C: 38081104
	v_add_nc_u32_e32 v6, v7, v6                                // 000000001B80: 4A0C0D07
	v_lshlrev_b32_e32 v5, 23, v6                               // 000000001B84: 300A0C97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001B88: BF870211
	v_sub_nc_u32_e32 v3, v3, v5                                // 000000001B8C: 4C060B03
	v_mul_f32_e32 v5, 0x3fc90fda, v4                           // 000000001B90: 100A08FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001B98: BF870112
	v_add_nc_u32_e32 v3, 0x33000000, v3                        // 000000001B9C: 4A0606FF 33000000
	v_fma_f32 v6, 0x3fc90fda, v4, -v5                          // 000000001BA4: D6130006 841608FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001BB0: BF870112
	v_or_b32_e32 v3, s8, v3                                    // 000000001BB4: 38060608
	v_fmac_f32_e32 v6, 0x33a22168, v4                          // 000000001BB8: 560C08FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC0: BF870091
	v_fmac_f32_e32 v6, 0x3fc90fda, v3                          // 000000001BC4: 560C06FF 3FC90FDA
	v_add_f32_e32 v3, v5, v6                                   // 000000001BCC: 06060D05
	s_cbranch_execz 2                                          // 000000001BD0: BFA50002 <E_2925n45+0x5dc>
	v_mov_b32_e32 v4, s7                                       // 000000001BD4: 7E080207
	s_branch 14                                                // 000000001BD8: BFA0000E <E_2925n45+0x614>
	v_mul_f32_e64 v3, 0x3f22f983, s6                           // 000000001BDC: D5080003 00000CFF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BE8: BF870091
	v_rndne_f32_e32 v4, v3                                     // 000000001BEC: 7E084703
	v_fma_f32 v3, 0xbfc90fda, v4, s6                           // 000000001BF0: D6130003 001A08FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BFC: BF870091
	v_fmac_f32_e32 v3, 0xb3a22168, v4                          // 000000001C00: 560608FF B3A22168
	v_fmac_f32_e32 v3, 0xa7c234c4, v4                          // 000000001C08: 560608FF A7C234C4
	v_cvt_i32_f32_e32 v4, v4                                   // 000000001C10: 7E081104
	v_dual_mul_f32 v5, v0, v0 :: v_dual_lshlrev_b32 v6, 30, v1 // 000000001C14: C8E20100 0506029E
	s_mov_b32 s7, 0xb94c1982                                   // 000000001C1C: BE8700FF B94C1982
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001C24: BF870123
	v_mul_f32_e32 v7, v3, v3                                   // 000000001C28: 100E0703
	s_mov_b32 s8, 0x37d75334                                   // 000000001C2C: BE8800FF 37D75334
	v_fmaak_f32 v8, s7, v5, 0x3c0881c4                         // 000000001C34: 5A100A07 3C0881C4
	v_and_b32_e32 v1, 1, v1                                    // 000000001C3C: 36020281
	s_xor_b32 s3, s2, s3                                       // 000000001C40: 8D030302
	v_fmaak_f32 v10, s7, v7, 0x3c0881c4                        // 000000001C44: 5A140E07 3C0881C4
	v_xor_b32_e32 v2, s6, v2                                   // 000000001C4C: 3A040406
	s_add_u32 s0, s0, s4                                       // 000000001C50: 80000400
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 000000001C54: 7C940280
	s_addc_u32 s1, s1, s5                                      // 000000001C58: 82010501
	v_fmaak_f32 v10, v7, v10, 0xbe2aaa9d                       // 000000001C5C: 5A141507 BE2AAA9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001C64: BF870121
	v_dual_fmaak_f32 v11, s8, v7, 0xbab64f3b :: v_dual_mul_f32 v10, v7, v10// 000000001C68: C8460E08 0B0A1507 BAB64F3B
	v_fmaak_f32 v8, v5, v8, 0xbe2aaa9d                         // 000000001C74: 5A101105 BE2AAA9D
	v_fmaak_f32 v11, v7, v11, 0x3d2aabf7                       // 000000001C7C: 5A161707 3D2AABF7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001C84: BF870193
	v_fmac_f32_e32 v3, v3, v10                                 // 000000001C88: 56061503
	v_dual_fmaak_f32 v9, s8, v5, 0xbab64f3b :: v_dual_mul_f32 v8, v5, v8// 000000001C8C: C8460A08 09081105 BAB64F3B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001C98: BF870121
	v_dual_fmaak_f32 v9, v5, v9, 0x3d2aabf7 :: v_dual_fmac_f32 v0, v0, v8// 000000001C9C: C8401305 09001100 3D2AABF7
	v_lshlrev_b32_e32 v8, 30, v4                               // 000000001CA8: 3010089E
	v_dual_fmaak_f32 v9, v5, v9, 0xbf000004 :: v_dual_and_b32 v4, 1, v4// 000000001CAC: C8641305 09040881 BF000004
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001CB8: BF870112
	v_and_b32_e32 v8, 0x80000000, v8                           // 000000001CBC: 361010FF 80000000
	v_fma_f32 v5, v5, v9, 1.0                                  // 000000001CC4: D6130005 03CA1305
	v_fmaak_f32 v11, v7, v11, 0xbf000004                       // 000000001CCC: 5A161707 BF000004
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001CD4: BF870193
	v_xor_b32_e32 v2, v2, v8                                   // 000000001CD8: 3A041102
	v_cndmask_b32_e32 v0, v5, v0, vcc_lo                       // 000000001CDC: 02000105
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001CE0: BF870123
	v_fma_f32 v7, v7, v11, 1.0                                 // 000000001CE4: D6130007 03CA1707
	v_cmp_eq_u32_e32 vcc_lo, 0, v4                             // 000000001CEC: 7C940880
	v_dual_cndmask_b32 v1, v7, v3 :: v_dual_and_b32 v6, 0x80000000, v6// 000000001CF0: CA640707 01060CFF 80000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001CFC: BF8701A1
	v_xor_b32_e32 v6, s3, v6                                   // 000000001D00: 3A0C0C03
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 000000001D04: D47E006A 0001FE02 000001F8
	v_xor_b32_e32 v1, v2, v1                                   // 000000001D10: 3A020302
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D14: BF870093
	v_xor_b32_e32 v0, v6, v0                                   // 000000001D18: 3A000106
	v_cndmask_b32_e32 v0, 0x7fc00000, v0, vcc_lo               // 000000001D1C: 020000FF 7FC00000
	v_cmp_class_f32_e64 vcc_lo, s6, 0x1f8                      // 000000001D24: D47E006A 0001FE06 000001F8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D30: BF870094
	v_cndmask_b32_e32 v1, 0x7fc00000, v1, vcc_lo               // 000000001D34: 020202FF 7FC00000
	v_div_scale_f32 v2, null, v1, v1, v0                       // 000000001D3C: D6FC7C02 04020301
	v_div_scale_f32 v5, vcc_lo, v0, v1, v0                     // 000000001D44: D6FC6A05 04020300
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001D4C: BF8700B2
	v_rcp_f32_e32 v3, v2                                       // 000000001D50: 7E065502
	s_waitcnt_depctr 0xfff                                     // 000000001D54: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 000000001D58: D6130004 23CA0702
	v_fmac_f32_e32 v3, v4, v3                                  // 000000001D60: 56060704
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D64: BF870091
	v_mul_f32_e32 v4, v5, v3                                   // 000000001D68: 10080705
	v_fma_f32 v6, -v2, v4, v5                                  // 000000001D6C: D6130006 24160902
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D74: BF870091
	v_fmac_f32_e32 v4, v6, v3                                  // 000000001D78: 56080706
	v_fma_f32 v2, -v2, v4, v5                                  // 000000001D7C: D6130002 24160902
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D84: BF870091
	v_div_fmas_f32 v2, v2, v3, v4                              // 000000001D88: D6370002 04120702
	v_div_fixup_f32 v0, v2, v1, v0                             // 000000001D90: D6270000 04020302
	v_mov_b32_e32 v1, 0                                        // 000000001D98: 7E020280
	global_store_b32 v1, v0, s[0:1]                            // 000000001D9C: DC6A0000 00000001
	s_nop 0                                                    // 000000001DA4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001DA8: BFB60003
	s_endpgm                                                   // 000000001DAC: BFB00000
