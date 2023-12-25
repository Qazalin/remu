
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n48>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[6:7], s[4:5], 3                               // 000000001614: 84868304
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b64 s[6:7], s[2:3], null                            // 000000001624: F4040181 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	s_clz_i32_u32 s2, s7                                       // 000000001630: BE820A07
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001634: BF870499
	s_min_u32 s8, s2, 32                                       // 000000001638: 8988A002
	s_lshl_b64 s[2:3], s[6:7], s8                              // 00000000163C: 84820806
	s_sub_i32 s8, 32, s8                                       // 000000001640: 818808A0
	s_min_u32 s2, s2, 1                                        // 000000001644: 89828102
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001648: BF8704C9
	s_or_b32 s2, s3, s2                                        // 00000000164C: 8C020203
	s_mov_b32 s3, 0                                            // 000000001650: BE830080
	v_cvt_f32_u32_e32 v0, s2                                   // 000000001654: 7E000C02
	s_mov_b32 s2, 0x1ffff                                      // 000000001658: BE8200FF 0001FFFF
	v_cmp_gt_u64_e64 s6, s[6:7], s[2:3]                        // 000000001660: D45C0006 00000406
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001668: BF870112
	v_ldexp_f32 v0, v0, s8                                     // 00000000166C: D71C0000 00001100
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001674: 8B6A067E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001678: BF870001
	v_readfirstlane_b32 s2, v0                                 // 00000000167C: 7E040500
	s_cbranch_vccz 165                                         // 000000001680: BFA300A5 <E_3n48+0x318>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001684: BF870001
	s_and_b32 s6, s2, 0x7fffff                                 // 000000001688: 8B06FF02 007FFFFF
	s_lshr_b32 s7, s2, 23                                      // 000000001690: 85079702
	s_bitset1_b32 s6, 23                                       // 000000001694: BE861297
	s_addk_i32 s7, 0xff88                                      // 000000001698: B787FF88
	s_mul_hi_u32 s8, s6, 0xfe5163ab                            // 00000000169C: 9688FF06 FE5163AB
	s_mul_i32 s9, s6, 0x3c439041                               // 0000000016A4: 9609FF06 3C439041
	s_mul_hi_u32 s10, s6, 0x3c439041                           // 0000000016AC: 968AFF06 3C439041
	s_add_u32 s8, s8, s9                                       // 0000000016B4: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000016B8: 82090A80
	s_mul_i32 s10, s6, 0xdb629599                              // 0000000016BC: 960AFF06 DB629599
	s_mul_hi_u32 s11, s6, 0xdb629599                           // 0000000016C4: 968BFF06 DB629599
	s_add_u32 s9, s9, s10                                      // 0000000016CC: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000016D0: 820A0B80
	s_mul_i32 s11, s6, 0xf534ddc0                              // 0000000016D4: 960BFF06 F534DDC0
	s_mul_hi_u32 s12, s6, 0xf534ddc0                           // 0000000016DC: 968CFF06 F534DDC0
	s_add_u32 s10, s10, s11                                    // 0000000016E4: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000016E8: 820B0C80
	s_mul_i32 s12, s6, 0xfc2757d1                              // 0000000016EC: 960CFF06 FC2757D1
	s_mul_hi_u32 s13, s6, 0xfc2757d1                           // 0000000016F4: 968DFF06 FC2757D1
	s_add_u32 s11, s11, s12                                    // 0000000016FC: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 000000001700: 820C0D80
	s_mul_i32 s13, s6, 0x4e441529                              // 000000001704: 960DFF06 4E441529
	s_mul_hi_u32 s14, s6, 0x4e441529                           // 00000000170C: 968EFF06 4E441529
	s_add_u32 s12, s12, s13                                    // 000000001714: 800C0D0C
	s_addc_u32 s13, 0, s14                                     // 000000001718: 820D0E80
	s_cmp_gt_u32 s7, 63                                        // 00000000171C: BF08BF07
	s_mul_i32 s14, s6, 0xfe5163ab                              // 000000001720: 960EFF06 FE5163AB
	s_mul_hi_u32 s15, s6, 0xa2f9836e                           // 000000001728: 968FFF06 A2F9836E
	s_mul_i32 s6, s6, 0xa2f9836e                               // 000000001730: 9606FF06 A2F9836E
	s_cselect_b32 s16, s9, s11                                 // 000000001738: 98100B09
	s_cselect_b32 s8, s8, s10                                  // 00000000173C: 98080A08
	s_cselect_b32 s9, s14, s9                                  // 000000001740: 9809090E
	s_add_u32 s6, s13, s6                                      // 000000001744: 8006060D
	s_addc_u32 s13, 0, s15                                     // 000000001748: 820D0F80
	s_cmp_gt_u32 s7, 63                                        // 00000000174C: BF08BF07
	s_cselect_b32 s14, 0xffffffc0, 0                           // 000000001750: 980E80FF FFFFFFC0
	s_cselect_b32 s10, s10, s12                                // 000000001758: 980A0C0A
	s_cselect_b32 s6, s11, s6                                  // 00000000175C: 9806060B
	s_cselect_b32 s11, s12, s13                                // 000000001760: 980B0D0C
	s_add_i32 s14, s14, s7                                     // 000000001764: 810E070E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001768: BF870009
	s_cmp_gt_u32 s14, 31                                       // 00000000176C: BF089F0E
	s_cselect_b32 s7, 0xffffffe0, 0                            // 000000001770: 980780FF FFFFFFE0
	s_cselect_b32 s12, s10, s6                                 // 000000001778: 980C060A
	s_cselect_b32 s6, s6, s11                                  // 00000000177C: 98060B06
	s_cselect_b32 s10, s16, s10                                // 000000001780: 980A0A10
	s_cselect_b32 s11, s8, s16                                 // 000000001784: 980B1008
	s_cselect_b32 s8, s9, s8                                   // 000000001788: 98080809
	s_add_i32 s7, s7, s14                                      // 00000000178C: 81070E07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001790: BF870009
	s_cmp_gt_u32 s7, 31                                        // 000000001794: BF089F07
	s_cselect_b32 s9, 0xffffffe0, 0                            // 000000001798: 980980FF FFFFFFE0
	s_cselect_b32 s6, s12, s6                                  // 0000000017A0: 9806060C
	s_cselect_b32 s12, s10, s12                                // 0000000017A4: 980C0C0A
	s_cselect_b32 s10, s11, s10                                // 0000000017A8: 980A0A0B
	s_cselect_b32 s8, s8, s11                                  // 0000000017AC: 98080B08
	s_add_i32 s9, s9, s7                                       // 0000000017B0: 81090709
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017B4: BF8700C9
	s_sub_i32 s7, 32, s9                                       // 0000000017B8: 818709A0
	s_cmp_eq_u32 s9, 0                                         // 0000000017BC: BF068009
	v_mov_b32_e32 v0, s7                                       // 0000000017C0: 7E000207
	s_cselect_b32 s9, -1, 0                                    // 0000000017C4: 980980C1
	v_alignbit_b32 v1, s6, s12, v0                             // 0000000017C8: D6160001 04001806
	v_alignbit_b32 v2, s12, s10, v0                            // 0000000017D0: D6160002 0400140C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017D8: BF870112
	v_readfirstlane_b32 s7, v1                                 // 0000000017DC: 7E0E0501
	v_cndmask_b32_e64 v1, v2, s12, s9                          // 0000000017E0: D5010001 00241902
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017E8: BF870002
	s_cselect_b32 s6, s6, s7                                   // 0000000017EC: 98060706
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017F0: BF870481
	v_alignbit_b32 v2, s6, v1, 30                              // 0000000017F4: D6160002 027A0206
	s_bfe_u32 s7, s6, 0x1001d                                  // 0000000017FC: 9307FF06 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001804: BF870009
	s_sub_i32 s11, 0, s7                                       // 000000001808: 818B0780
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 00000000180C: BF870481
	v_xor_b32_e32 v2, s11, v2                                  // 000000001810: 3A04040B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001814: BF870091
	v_clz_i32_u32_e32 v3, v2                                   // 000000001818: 7E067302
	v_min_u32_e32 v3, 32, v3                                   // 00000000181C: 260606A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001820: BF870131
	v_lshlrev_b32_e32 v5, 23, v3                               // 000000001824: 300A0697
	v_alignbit_b32 v0, s10, s8, v0                             // 000000001828: D6160000 0400100A
	v_sub_nc_u32_e32 v4, 31, v3                                // 000000001830: 4C08069F
	v_cndmask_b32_e64 v0, v0, s10, s9                          // 000000001834: D5010000 00241500
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000183C: BF870001
	v_alignbit_b32 v1, v1, v0, 30                              // 000000001840: D6160001 027A0101
	v_alignbit_b32 v0, v0, s8, 30                              // 000000001848: D6160000 02781100
	s_lshr_b32 s8, s6, 29                                      // 000000001850: 85089D06
	s_lshr_b32 s6, s6, 30                                      // 000000001854: 85069E06
	s_lshl_b32 s8, s8, 31                                      // 000000001858: 84089F08
	v_xor_b32_e32 v1, s11, v1                                  // 00000000185C: 3A02020B
	v_xor_b32_e32 v0, s11, v0                                  // 000000001860: 3A00000B
	s_or_b32 s9, s8, 0.5                                       // 000000001864: 8C09F008
	s_add_i32 s6, s7, s6                                       // 000000001868: 81060607
	v_sub_nc_u32_e32 v5, s9, v5                                // 00000000186C: 4C0A0A09
	v_alignbit_b32 v2, v2, v1, v4                              // 000000001870: D6160002 04120302
	v_alignbit_b32 v0, v1, v0, v4                              // 000000001878: D6160000 04120101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001880: BF870091
	v_alignbit_b32 v1, v2, v0, 9                               // 000000001884: D6160001 02260102
	v_clz_i32_u32_e32 v4, v1                                   // 00000000188C: 7E087301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001890: BF870091
	v_min_u32_e32 v4, 32, v4                                   // 000000001894: 260808A0
	v_sub_nc_u32_e32 v6, 31, v4                                // 000000001898: 4C0C089F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000189C: BF870121
	v_alignbit_b32 v0, v1, v0, v6                              // 0000000018A0: D6160000 041A0101
	v_lshrrev_b32_e32 v1, 9, v2                                // 0000000018A8: 32020489
	v_lshrrev_b32_e32 v0, 9, v0                                // 0000000018AC: 32000089
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018B0: BF8700A2
	v_or_b32_e32 v1, v1, v5                                    // 0000000018B4: 38020B01
	v_add_nc_u32_e32 v3, v4, v3                                // 0000000018B8: 4A060704
	v_lshlrev_b32_e32 v2, 23, v3                               // 0000000018BC: 30040697
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000018C0: BF870211
	v_sub_nc_u32_e32 v0, v0, v2                                // 0000000018C4: 4C000500
	v_mul_f32_e32 v2, 0x3fc90fda, v1                           // 0000000018C8: 100402FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018D0: BF870112
	v_add_nc_u32_e32 v0, 0x33000000, v0                        // 0000000018D4: 4A0000FF 33000000
	v_fma_f32 v3, 0x3fc90fda, v1, -v2                          // 0000000018DC: D6130003 840A02FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018E8: BF870112
	v_or_b32_e32 v0, s8, v0                                    // 0000000018EC: 38000008
	v_fmac_f32_e32 v3, 0x33a22168, v1                          // 0000000018F0: 560602FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F8: BF870091
	v_fmac_f32_e32 v3, 0x3fc90fda, v0                          // 0000000018FC: 560600FF 3FC90FDA
	v_add_f32_e32 v0, v2, v3                                   // 000000001904: 06000702
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 000000001908: 916A037E
	s_cbranch_vccz 2                                           // 00000000190C: BFA30002 <E_3n48+0x318>
	v_mov_b32_e32 v1, s6                                       // 000000001910: 7E020206
	s_branch 15                                                // 000000001914: BFA0000F <E_3n48+0x354>
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001918: BF870091
	v_mul_f32_e64 v0, 0x3f22f983, s2                           // 00000000191C: D5080000 000004FF 3F22F983
	v_rndne_f32_e32 v1, v0                                     // 000000001928: 7E024700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000192C: BF870091
	v_fma_f32 v0, 0xbfc90fda, v1, s2                           // 000000001930: D6130000 000A02FF BFC90FDA
	v_fmac_f32_e32 v0, 0xb3a22168, v1                          // 00000000193C: 560002FF B3A22168
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001944: BF870001
	v_fmac_f32_e32 v0, 0xa7c234c4, v1                          // 000000001948: 560002FF A7C234C4
	v_cvt_i32_f32_e32 v1, v1                                   // 000000001950: 7E021101
	v_sub_f32_e64 v2, 0x3fc90fdb, s2                           // 000000001954: D5040002 000004FF 3FC90FDB
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001960: BF870121
	v_cmp_ngt_f32_e64 s3, 0x48000000, |v2|                     // 000000001964: D41B0203 000204FF 48000000
	v_readfirstlane_b32 s6, v2                                 // 000000001970: 7E0C0502
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001974: 8B6A037E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001978: BF870001
	s_and_b32 s3, s6, 0x7fffffff                               // 00000000197C: 8B03FF06 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001984: BFA300A4 <E_3n48+0x618>
	s_and_b32 s6, s3, 0x7fffff                                 // 000000001988: 8B06FF03 007FFFFF
	s_lshr_b32 s7, s3, 23                                      // 000000001990: 85079703
	s_bitset1_b32 s6, 23                                       // 000000001994: BE861297
	s_addk_i32 s7, 0xff88                                      // 000000001998: B787FF88
	s_mul_hi_u32 s8, s6, 0xfe5163ab                            // 00000000199C: 9688FF06 FE5163AB
	s_mul_i32 s9, s6, 0x3c439041                               // 0000000019A4: 9609FF06 3C439041
	s_mul_hi_u32 s10, s6, 0x3c439041                           // 0000000019AC: 968AFF06 3C439041
	s_add_u32 s8, s8, s9                                       // 0000000019B4: 80080908
	s_addc_u32 s9, 0, s10                                      // 0000000019B8: 82090A80
	s_mul_i32 s10, s6, 0xdb629599                              // 0000000019BC: 960AFF06 DB629599
	s_mul_hi_u32 s11, s6, 0xdb629599                           // 0000000019C4: 968BFF06 DB629599
	s_add_u32 s9, s9, s10                                      // 0000000019CC: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000019D0: 820A0B80
	s_mul_i32 s11, s6, 0xf534ddc0                              // 0000000019D4: 960BFF06 F534DDC0
	s_mul_hi_u32 s12, s6, 0xf534ddc0                           // 0000000019DC: 968CFF06 F534DDC0
	s_add_u32 s10, s10, s11                                    // 0000000019E4: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000019E8: 820B0C80
	s_mul_i32 s12, s6, 0xfc2757d1                              // 0000000019EC: 960CFF06 FC2757D1
	s_mul_hi_u32 s13, s6, 0xfc2757d1                           // 0000000019F4: 968DFF06 FC2757D1
	s_add_u32 s11, s11, s12                                    // 0000000019FC: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 000000001A00: 820C0D80
	s_mul_i32 s13, s6, 0x4e441529                              // 000000001A04: 960DFF06 4E441529
	s_mul_hi_u32 s14, s6, 0x4e441529                           // 000000001A0C: 968EFF06 4E441529
	s_add_u32 s12, s12, s13                                    // 000000001A14: 800C0D0C
	s_addc_u32 s13, 0, s14                                     // 000000001A18: 820D0E80
	s_cmp_gt_u32 s7, 63                                        // 000000001A1C: BF08BF07
	s_mul_i32 s14, s6, 0xfe5163ab                              // 000000001A20: 960EFF06 FE5163AB
	s_mul_hi_u32 s15, s6, 0xa2f9836e                           // 000000001A28: 968FFF06 A2F9836E
	s_mul_i32 s6, s6, 0xa2f9836e                               // 000000001A30: 9606FF06 A2F9836E
	s_cselect_b32 s16, s9, s11                                 // 000000001A38: 98100B09
	s_cselect_b32 s8, s8, s10                                  // 000000001A3C: 98080A08
	s_cselect_b32 s9, s14, s9                                  // 000000001A40: 9809090E
	s_add_u32 s6, s13, s6                                      // 000000001A44: 8006060D
	s_addc_u32 s13, 0, s15                                     // 000000001A48: 820D0F80
	s_cmp_gt_u32 s7, 63                                        // 000000001A4C: BF08BF07
	s_cselect_b32 s14, 0xffffffc0, 0                           // 000000001A50: 980E80FF FFFFFFC0
	s_cselect_b32 s10, s10, s12                                // 000000001A58: 980A0C0A
	s_cselect_b32 s6, s11, s6                                  // 000000001A5C: 9806060B
	s_cselect_b32 s11, s12, s13                                // 000000001A60: 980B0D0C
	s_add_i32 s14, s14, s7                                     // 000000001A64: 810E070E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A68: BF870009
	s_cmp_gt_u32 s14, 31                                       // 000000001A6C: BF089F0E
	s_cselect_b32 s7, 0xffffffe0, 0                            // 000000001A70: 980780FF FFFFFFE0
	s_cselect_b32 s12, s10, s6                                 // 000000001A78: 980C060A
	s_cselect_b32 s6, s6, s11                                  // 000000001A7C: 98060B06
	s_cselect_b32 s10, s16, s10                                // 000000001A80: 980A0A10
	s_cselect_b32 s11, s8, s16                                 // 000000001A84: 980B1008
	s_cselect_b32 s8, s9, s8                                   // 000000001A88: 98080809
	s_add_i32 s7, s7, s14                                      // 000000001A8C: 81070E07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A90: BF870009
	s_cmp_gt_u32 s7, 31                                        // 000000001A94: BF089F07
	s_cselect_b32 s9, 0xffffffe0, 0                            // 000000001A98: 980980FF FFFFFFE0
	s_cselect_b32 s6, s12, s6                                  // 000000001AA0: 9806060C
	s_cselect_b32 s12, s10, s12                                // 000000001AA4: 980C0C0A
	s_cselect_b32 s10, s11, s10                                // 000000001AA8: 980A0A0B
	s_cselect_b32 s8, s8, s11                                  // 000000001AAC: 98080B08
	s_add_i32 s9, s9, s7                                       // 000000001AB0: 81090709
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001AB4: BF8700C9
	s_sub_i32 s7, 32, s9                                       // 000000001AB8: 818709A0
	s_cmp_eq_u32 s9, 0                                         // 000000001ABC: BF068009
	v_mov_b32_e32 v3, s7                                       // 000000001AC0: 7E060207
	s_cselect_b32 s9, -1, 0                                    // 000000001AC4: 980980C1
	v_alignbit_b32 v4, s6, s12, v3                             // 000000001AC8: D6160004 040C1806
	v_alignbit_b32 v5, s12, s10, v3                            // 000000001AD0: D6160005 040C140C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001AD8: BF870112
	v_readfirstlane_b32 s7, v4                                 // 000000001ADC: 7E0E0504
	v_cndmask_b32_e64 v4, v5, s12, s9                          // 000000001AE0: D5010004 00241905
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001AE8: BF870002
	s_cselect_b32 s6, s6, s7                                   // 000000001AEC: 98060706
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001AF0: BF870481
	v_alignbit_b32 v5, s6, v4, 30                              // 000000001AF4: D6160005 027A0806
	s_bfe_u32 s11, s6, 0x1001d                                 // 000000001AFC: 930BFF06 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B04: BF870009
	s_sub_i32 s7, 0, s11                                       // 000000001B08: 81870B80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001B0C: BF870481
	v_xor_b32_e32 v5, s7, v5                                   // 000000001B10: 3A0A0A07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B14: BF870091
	v_clz_i32_u32_e32 v6, v5                                   // 000000001B18: 7E0C7305
	v_min_u32_e32 v6, 32, v6                                   // 000000001B1C: 260C0CA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001B20: BF870131
	v_lshlrev_b32_e32 v8, 23, v6                               // 000000001B24: 30100C97
	v_alignbit_b32 v3, s10, s8, v3                             // 000000001B28: D6160003 040C100A
	v_sub_nc_u32_e32 v7, 31, v6                                // 000000001B30: 4C0E0C9F
	v_cndmask_b32_e64 v3, v3, s10, s9                          // 000000001B34: D5010003 00241503
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001B3C: BF870121
	v_alignbit_b32 v4, v4, v3, 30                              // 000000001B40: D6160004 027A0704
	v_alignbit_b32 v3, v3, s8, 30                              // 000000001B48: D6160003 02781103
	v_xor_b32_e32 v4, s7, v4                                   // 000000001B50: 3A080807
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B54: BF870002
	v_xor_b32_e32 v3, s7, v3                                   // 000000001B58: 3A060607
	s_lshr_b32 s7, s6, 29                                      // 000000001B5C: 85079D06
	s_lshr_b32 s6, s6, 30                                      // 000000001B60: 85069E06
	s_lshl_b32 s7, s7, 31                                      // 000000001B64: 84079F07
	v_alignbit_b32 v5, v5, v4, v7                              // 000000001B68: D6160005 041E0905
	v_alignbit_b32 v3, v4, v3, v7                              // 000000001B70: D6160003 041E0704
	s_or_b32 s8, s7, 0.5                                       // 000000001B78: 8C08F007
	s_add_i32 s6, s11, s6                                      // 000000001B7C: 8106060B
	v_sub_nc_u32_e32 v8, s8, v8                                // 000000001B80: 4C101008
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B84: BF870092
	v_alignbit_b32 v4, v5, v3, 9                               // 000000001B88: D6160004 02260705
	v_clz_i32_u32_e32 v7, v4                                   // 000000001B90: 7E0E7304
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B94: BF870091
	v_min_u32_e32 v7, 32, v7                                   // 000000001B98: 260E0EA0
	v_sub_nc_u32_e32 v9, 31, v7                                // 000000001B9C: 4C120E9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001BA0: BF870121
	v_alignbit_b32 v3, v4, v3, v9                              // 000000001BA4: D6160003 04260704
	v_lshrrev_b32_e32 v4, 9, v5                                // 000000001BAC: 32080A89
	v_lshrrev_b32_e32 v3, 9, v3                                // 000000001BB0: 32060689
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001BB4: BF8700A2
	v_or_b32_e32 v4, v4, v8                                    // 000000001BB8: 38081104
	v_add_nc_u32_e32 v6, v7, v6                                // 000000001BBC: 4A0C0D07
	v_lshlrev_b32_e32 v5, 23, v6                               // 000000001BC0: 300A0C97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001BC4: BF870211
	v_sub_nc_u32_e32 v3, v3, v5                                // 000000001BC8: 4C060B03
	v_mul_f32_e32 v5, 0x3fc90fda, v4                           // 000000001BCC: 100A08FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001BD4: BF870112
	v_add_nc_u32_e32 v3, 0x33000000, v3                        // 000000001BD8: 4A0606FF 33000000
	v_fma_f32 v6, 0x3fc90fda, v4, -v5                          // 000000001BE0: D6130006 841608FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001BEC: BF870112
	v_or_b32_e32 v3, s7, v3                                    // 000000001BF0: 38060607
	v_fmac_f32_e32 v6, 0x33a22168, v4                          // 000000001BF4: 560C08FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BFC: BF870091
	v_fmac_f32_e32 v6, 0x3fc90fda, v3                          // 000000001C00: 560C06FF 3FC90FDA
	v_add_f32_e32 v3, v5, v6                                   // 000000001C08: 06060D05
	s_cbranch_execz 2                                          // 000000001C0C: BFA50002 <E_3n48+0x618>
	v_mov_b32_e32 v4, s6                                       // 000000001C10: 7E080206
	s_branch 14                                                // 000000001C14: BFA0000E <E_3n48+0x650>
	v_mul_f32_e64 v3, 0x3f22f983, s3                           // 000000001C18: D5080003 000006FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C24: BF870091
	v_rndne_f32_e32 v4, v3                                     // 000000001C28: 7E084703
	v_fma_f32 v3, 0xbfc90fda, v4, s3                           // 000000001C2C: D6130003 000E08FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C38: BF870091
	v_fmac_f32_e32 v3, 0xb3a22168, v4                          // 000000001C3C: 560608FF B3A22168
	v_fmac_f32_e32 v3, 0xa7c234c4, v4                          // 000000001C44: 560608FF A7C234C4
	v_cvt_i32_f32_e32 v4, v4                                   // 000000001C4C: 7E081104
	v_dual_mul_f32 v5, v0, v0 :: v_dual_lshlrev_b32 v6, 30, v1 // 000000001C50: C8E20100 0506029E
	s_mov_b32 s7, 0xb94c1982                                   // 000000001C58: BE8700FF B94C1982
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001C60: BF870123
	v_mul_f32_e32 v7, v3, v3                                   // 000000001C64: 100E0703
	s_mov_b32 s8, 0x37d75334                                   // 000000001C68: BE8800FF 37D75334
	v_fmaak_f32 v8, s7, v5, 0x3c0881c4                         // 000000001C70: 5A100A07 3C0881C4
	v_and_b32_e32 v1, 1, v1                                    // 000000001C78: 36020281
	s_xor_b32 s6, s2, s2                                       // 000000001C7C: 8D060202
	v_fmaak_f32 v10, s7, v7, 0x3c0881c4                        // 000000001C80: 5A140E07 3C0881C4
	v_xor_b32_e32 v2, s3, v2                                   // 000000001C88: 3A040403
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001C8C: BF870193
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 000000001C90: 7C940280
	v_fmaak_f32 v10, v7, v10, 0xbe2aaa9d                       // 000000001C94: 5A141507 BE2AAA9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001C9C: BF870121
	v_dual_fmaak_f32 v11, s8, v7, 0xbab64f3b :: v_dual_mul_f32 v10, v7, v10// 000000001CA0: C8460E08 0B0A1507 BAB64F3B
	v_fmaak_f32 v8, v5, v8, 0xbe2aaa9d                         // 000000001CAC: 5A101105 BE2AAA9D
	v_fmaak_f32 v11, v7, v11, 0x3d2aabf7                       // 000000001CB4: 5A161707 3D2AABF7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001CBC: BF870193
	v_fmac_f32_e32 v3, v3, v10                                 // 000000001CC0: 56061503
	v_dual_fmaak_f32 v9, s8, v5, 0xbab64f3b :: v_dual_mul_f32 v8, v5, v8// 000000001CC4: C8460A08 09081105 BAB64F3B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001CD0: BF870121
	v_dual_fmaak_f32 v9, v5, v9, 0x3d2aabf7 :: v_dual_fmac_f32 v0, v0, v8// 000000001CD4: C8401305 09001100 3D2AABF7
	v_lshlrev_b32_e32 v8, 30, v4                               // 000000001CE0: 3010089E
	v_dual_fmaak_f32 v9, v5, v9, 0xbf000004 :: v_dual_and_b32 v4, 1, v4// 000000001CE4: C8641305 09040881 BF000004
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001CF0: BF870112
	v_and_b32_e32 v8, 0x80000000, v8                           // 000000001CF4: 361010FF 80000000
	v_fma_f32 v5, v5, v9, 1.0                                  // 000000001CFC: D6130005 03CA1305
	v_fmaak_f32 v11, v7, v11, 0xbf000004                       // 000000001D04: 5A161707 BF000004
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001D0C: BF870193
	v_xor_b32_e32 v2, v2, v8                                   // 000000001D10: 3A041102
	v_cndmask_b32_e32 v0, v5, v0, vcc_lo                       // 000000001D14: 02000105
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001D18: BF870123
	v_fma_f32 v7, v7, v11, 1.0                                 // 000000001D1C: D6130007 03CA1707
	v_cmp_eq_u32_e32 vcc_lo, 0, v4                             // 000000001D24: 7C940880
	v_dual_cndmask_b32 v1, v7, v3 :: v_dual_and_b32 v6, 0x80000000, v6// 000000001D28: CA640707 01060CFF 80000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001D34: BF8701A1
	v_xor_b32_e32 v6, s6, v6                                   // 000000001D38: 3A0C0C06
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 000000001D3C: D47E006A 0001FE02 000001F8
	v_xor_b32_e32 v1, v2, v1                                   // 000000001D48: 3A020302
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D4C: BF870093
	v_xor_b32_e32 v0, v6, v0                                   // 000000001D50: 3A000106
	v_cndmask_b32_e32 v0, 0x7fc00000, v0, vcc_lo               // 000000001D54: 020000FF 7FC00000
	v_cmp_class_f32_e64 vcc_lo, s3, 0x1f8                      // 000000001D5C: D47E006A 0001FE03 000001F8
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001D68: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001D6C: BF8700B9
	s_add_u32 s0, s0, s2                                       // 000000001D70: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001D74: 82010301
	v_cndmask_b32_e32 v1, 0x7fc00000, v1, vcc_lo               // 000000001D78: 020202FF 7FC00000
	v_div_scale_f32 v2, null, v1, v1, v0                       // 000000001D80: D6FC7C02 04020301
	v_div_scale_f32 v5, vcc_lo, v0, v1, v0                     // 000000001D88: D6FC6A05 04020300
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001D90: BF8700B2
	v_rcp_f32_e32 v3, v2                                       // 000000001D94: 7E065502
	s_waitcnt_depctr 0xfff                                     // 000000001D98: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 000000001D9C: D6130004 23CA0702
	v_fmac_f32_e32 v3, v4, v3                                  // 000000001DA4: 56060704
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DA8: BF870091
	v_mul_f32_e32 v4, v5, v3                                   // 000000001DAC: 10080705
	v_fma_f32 v6, -v2, v4, v5                                  // 000000001DB0: D6130006 24160902
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DB8: BF870091
	v_fmac_f32_e32 v4, v6, v3                                  // 000000001DBC: 56080706
	v_fma_f32 v2, -v2, v4, v5                                  // 000000001DC0: D6130002 24160902
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DC8: BF870091
	v_div_fmas_f32 v2, v2, v3, v4                              // 000000001DCC: D6370002 04120702
	v_div_fixup_f32 v0, v2, v1, v0                             // 000000001DD4: D6270000 04020302
	v_mov_b32_e32 v1, 0                                        // 000000001DDC: 7E020280
	global_store_b32 v1, v0, s[0:1]                            // 000000001DE0: DC6A0000 00000001
	s_nop 0                                                    // 000000001DE8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001DEC: BFB60003
	s_endpgm                                                   // 000000001DF0: BFB00000
