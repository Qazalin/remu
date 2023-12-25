
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n40>:
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
	v_cmp_ngt_f32_e64 s3, 0x48000000, |s2|                     // 000000001630: D41B0203 000004FF 48000000
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000163C: BF870001
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001640: 8B6A037E
	s_and_b32 s3, s2, 0x7fffffff                               // 000000001644: 8B03FF02 7FFFFFFF
	s_cbranch_vccz 164                                         // 00000000164C: BFA300A4 <E_2925n40+0x2e0>
	s_and_b32 s6, s3, 0x7fffff                                 // 000000001650: 8B06FF03 007FFFFF
	s_lshr_b32 s7, s3, 23                                      // 000000001658: 85079703
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
	s_cbranch_execz 2                                          // 0000000018D4: BFA50002 <E_2925n40+0x2e0>
	v_mov_b32_e32 v1, s6                                       // 0000000018D8: 7E020206
	s_branch 14                                                // 0000000018DC: BFA0000E <E_2925n40+0x318>
	v_mul_f32_e64 v0, 0x3f22f983, s3                           // 0000000018E0: D5080000 000006FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018EC: BF870091
	v_rndne_f32_e32 v1, v0                                     // 0000000018F0: 7E024700
	v_fma_f32 v0, 0xbfc90fda, v1, s3                           // 0000000018F4: D6130000 000E02FF BFC90FDA
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001900: BF870091
	v_fmac_f32_e32 v0, 0xb3a22168, v1                          // 000000001904: 560002FF B3A22168
	v_fmac_f32_e32 v0, 0xa7c234c4, v1                          // 00000000190C: 560002FF A7C234C4
	v_cvt_i32_f32_e32 v1, v1                                   // 000000001914: 7E021101
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001918: BF8700C1
	v_dual_mul_f32 v2, v0, v0 :: v_dual_lshlrev_b32 v5, 30, v1 // 00000000191C: C8E20100 0204029E
	s_mov_b32 s6, 0xb94c1982                                   // 000000001924: BE8600FF B94C1982
	s_mov_b32 s7, 0x37d75334                                   // 00000000192C: BE8700FF 37D75334
	s_xor_b32 s2, s3, s2                                       // 000000001934: 8D020203
	v_fmaak_f32 v3, s6, v2, 0x3c0881c4                         // 000000001938: 5A060406 3C0881C4
	s_add_u32 s0, s0, s4                                       // 000000001940: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001944: 82010501
	v_and_b32_e32 v1, 1, v1                                    // 000000001948: 36020281
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 00000000194C: BF8701A2
	v_fmaak_f32 v3, v2, v3, 0xbe2aaa9d                         // 000000001950: 5A060702 BE2AAA9D
	v_fmaak_f32 v4, s7, v2, 0xbab64f3b                         // 000000001958: 5A080407 BAB64F3B
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 000000001960: 7C940280
	v_mov_b32_e32 v1, 0                                        // 000000001964: 7E020280
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001968: BF870214
	v_mul_f32_e32 v3, v2, v3                                   // 00000000196C: 10060702
	v_fmaak_f32 v4, v2, v4, 0x3d2aabf7                         // 000000001970: 5A080902 3D2AABF7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001978: BF870112
	v_dual_fmac_f32 v0, v0, v3 :: v_dual_and_b32 v5, 0x80000000, v5// 00000000197C: C8240700 00040AFF 80000000
	v_fmaak_f32 v4, v2, v4, 0xbf000004                         // 000000001988: 5A080902 BF000004
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001990: BF870112
	v_xor_b32_e32 v3, s2, v5                                   // 000000001994: 3A060A02
	v_fma_f32 v2, v2, v4, 1.0                                  // 000000001998: D6130002 03CA0902
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000019A0: BF870121
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo                       // 0000000019A4: 02000102
	v_cmp_class_f32_e64 vcc_lo, s3, 0x1f8                      // 0000000019A8: D47E006A 0001FE03 000001F8
	v_xor_b32_e32 v0, v3, v0                                   // 0000000019B4: 3A000103
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019B8: BF870001
	v_cndmask_b32_e32 v0, 0x7fc00000, v0, vcc_lo               // 0000000019BC: 020000FF 7FC00000
	global_store_b32 v1, v0, s[0:1]                            // 0000000019C4: DC6A0000 00000001
	s_nop 0                                                    // 0000000019CC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000019D0: BFB60003
	s_endpgm                                                   // 0000000019D4: BFB00000
