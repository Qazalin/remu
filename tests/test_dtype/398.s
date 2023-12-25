
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n46>:
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
	s_cbranch_vccz 165                                         // 000000001680: BFA300A5 <E_3n46+0x318>
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
	s_cbranch_vccz 2                                           // 00000000190C: BFA30002 <E_3n46+0x318>
	v_mov_b32_e32 v1, s6                                       // 000000001910: 7E020206
	s_branch 15                                                // 000000001914: BFA0000F <E_3n46+0x354>
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001918: BF870091
	v_mul_f32_e64 v0, 0x3f22f983, s2                           // 00000000191C: D5080000 000004FF 3F22F983
	v_rndne_f32_e32 v1, v0                                     // 000000001928: 7E024700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000192C: BF870091
	v_fma_f32 v0, 0xbfc90fda, v1, s2                           // 000000001930: D6130000 000A02FF BFC90FDA
	v_fmac_f32_e32 v0, 0xb3a22168, v1                          // 00000000193C: 560002FF B3A22168
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001944: BF870001
	v_fmac_f32_e32 v0, 0xa7c234c4, v1                          // 000000001948: 560002FF A7C234C4
	v_cvt_i32_f32_e32 v1, v1                                   // 000000001950: 7E021101
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001954: BF8700C1
	v_dual_mul_f32 v2, v0, v0 :: v_dual_and_b32 v5, 1, v1      // 000000001958: C8E40100 02040281
	s_mov_b32 s3, 0xb94c1982                                   // 000000001960: BE8300FF B94C1982
	s_mov_b32 s6, 0x37d75334                                   // 000000001968: BE8600FF 37D75334
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001970: 84848204
	v_fmaak_f32 v3, s3, v2, 0x3c0881c4                         // 000000001974: 5A060403 3C0881C4
	v_cmp_eq_u32_e32 vcc_lo, 0, v5                             // 00000000197C: 7C940A80
	s_add_u32 s0, s0, s4                                       // 000000001980: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001984: 82010501
	v_lshlrev_b32_e32 v1, 30, v1                               // 000000001988: 3002029E
	v_fmaak_f32 v3, v2, v3, 0xbe2aaa9d                         // 00000000198C: 5A060702 BE2AAA9D
	v_fmaak_f32 v4, s6, v2, 0xbab64f3b                         // 000000001994: 5A080406 BAB64F3B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000199C: BF870193
	v_and_b32_e32 v1, 0x80000000, v1                           // 0000000019A0: 360202FF 80000000
	v_mul_f32_e32 v3, v2, v3                                   // 0000000019A8: 10060702
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000019AC: BF870113
	v_fmaak_f32 v4, v2, v4, 0x3d2aabf7                         // 0000000019B0: 5A080902 3D2AABF7
	v_fmac_f32_e32 v0, v0, v3                                  // 0000000019B8: 56000700
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019BC: BF870092
	v_fmaak_f32 v4, v2, v4, 0xbf000004                         // 0000000019C0: 5A080902 BF000004
	v_fma_f32 v2, v2, v4, 1.0                                  // 0000000019C8: D6130002 03CA0902
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000019D0: BF870121
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo                       // 0000000019D4: 02000102
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 0000000019D8: D47E006A 0001FE02 000001F8
	v_xor_b32_e32 v0, v1, v0                                   // 0000000019E4: 3A000101
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019E8: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_cndmask_b32 v0, 0x7fc00000, v0// 0000000019EC: CA120080 010000FF 7FC00000
	global_store_b32 v1, v0, s[0:1]                            // 0000000019F8: DC6A0000 00000001
	s_nop 0                                                    // 000000001A00: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A04: BFB60003
	s_endpgm                                                   // 000000001A08: BFB00000
