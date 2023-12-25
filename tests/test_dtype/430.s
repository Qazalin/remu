
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n78>:
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
	v_cvt_f32_u32_e32 v0, s3                                   // 000000001630: 7E000C03
	s_cmp_gt_u32 s3, 0x1ffff                                   // 000000001634: BF08FF03 0001FFFF
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000163C: BF870001
	v_readfirstlane_b32 s2, v0                                 // 000000001640: 7E040500
	s_cbranch_scc0 165                                         // 000000001644: BFA100A5 <E_3n78+0x2dc>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001648: BF870001
	s_and_b32 s3, s2, 0x7fffff                                 // 00000000164C: 8B03FF02 007FFFFF
	s_lshr_b32 s6, s2, 23                                      // 000000001654: 85069702
	s_bitset1_b32 s3, 23                                       // 000000001658: BE831297
	s_addk_i32 s6, 0xff88                                      // 00000000165C: B786FF88
	s_mul_hi_u32 s7, s3, 0xfe5163ab                            // 000000001660: 9687FF03 FE5163AB
	s_mul_i32 s8, s3, 0x3c439041                               // 000000001668: 9608FF03 3C439041
	s_mul_hi_u32 s9, s3, 0x3c439041                            // 000000001670: 9689FF03 3C439041
	s_add_u32 s7, s7, s8                                       // 000000001678: 80070807
	s_addc_u32 s8, 0, s9                                       // 00000000167C: 82080980
	s_mul_i32 s9, s3, 0xdb629599                               // 000000001680: 9609FF03 DB629599
	s_mul_hi_u32 s10, s3, 0xdb629599                           // 000000001688: 968AFF03 DB629599
	s_add_u32 s8, s8, s9                                       // 000000001690: 80080908
	s_addc_u32 s9, 0, s10                                      // 000000001694: 82090A80
	s_mul_i32 s10, s3, 0xf534ddc0                              // 000000001698: 960AFF03 F534DDC0
	s_mul_hi_u32 s11, s3, 0xf534ddc0                           // 0000000016A0: 968BFF03 F534DDC0
	s_add_u32 s9, s9, s10                                      // 0000000016A8: 80090A09
	s_addc_u32 s10, 0, s11                                     // 0000000016AC: 820A0B80
	s_mul_i32 s11, s3, 0xfc2757d1                              // 0000000016B0: 960BFF03 FC2757D1
	s_mul_hi_u32 s12, s3, 0xfc2757d1                           // 0000000016B8: 968CFF03 FC2757D1
	s_add_u32 s10, s10, s11                                    // 0000000016C0: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000016C4: 820B0C80
	s_mul_i32 s12, s3, 0x4e441529                              // 0000000016C8: 960CFF03 4E441529
	s_mul_hi_u32 s13, s3, 0x4e441529                           // 0000000016D0: 968DFF03 4E441529
	s_add_u32 s11, s11, s12                                    // 0000000016D8: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000016DC: 820C0D80
	s_cmp_gt_u32 s6, 63                                        // 0000000016E0: BF08BF06
	s_mul_i32 s13, s3, 0xfe5163ab                              // 0000000016E4: 960DFF03 FE5163AB
	s_mul_hi_u32 s14, s3, 0xa2f9836e                           // 0000000016EC: 968EFF03 A2F9836E
	s_mul_i32 s3, s3, 0xa2f9836e                               // 0000000016F4: 9603FF03 A2F9836E
	s_cselect_b32 s15, s8, s10                                 // 0000000016FC: 980F0A08
	s_cselect_b32 s7, s7, s9                                   // 000000001700: 98070907
	s_cselect_b32 s8, s13, s8                                  // 000000001704: 9808080D
	s_add_u32 s3, s12, s3                                      // 000000001708: 8003030C
	s_addc_u32 s12, 0, s14                                     // 00000000170C: 820C0E80
	s_cmp_gt_u32 s6, 63                                        // 000000001710: BF08BF06
	s_cselect_b32 s13, 0xffffffc0, 0                           // 000000001714: 980D80FF FFFFFFC0
	s_cselect_b32 s9, s9, s11                                  // 00000000171C: 98090B09
	s_cselect_b32 s3, s10, s3                                  // 000000001720: 9803030A
	s_cselect_b32 s10, s11, s12                                // 000000001724: 980A0C0B
	s_add_i32 s13, s13, s6                                     // 000000001728: 810D060D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000172C: BF870009
	s_cmp_gt_u32 s13, 31                                       // 000000001730: BF089F0D
	s_cselect_b32 s6, 0xffffffe0, 0                            // 000000001734: 980680FF FFFFFFE0
	s_cselect_b32 s11, s9, s3                                  // 00000000173C: 980B0309
	s_cselect_b32 s3, s3, s10                                  // 000000001740: 98030A03
	s_cselect_b32 s9, s15, s9                                  // 000000001744: 9809090F
	s_cselect_b32 s10, s7, s15                                 // 000000001748: 980A0F07
	s_cselect_b32 s7, s8, s7                                   // 00000000174C: 98070708
	s_add_i32 s6, s6, s13                                      // 000000001750: 81060D06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_cmp_gt_u32 s6, 31                                        // 000000001758: BF089F06
	s_cselect_b32 s8, 0xffffffe0, 0                            // 00000000175C: 980880FF FFFFFFE0
	s_cselect_b32 s3, s11, s3                                  // 000000001764: 9803030B
	s_cselect_b32 s11, s9, s11                                 // 000000001768: 980B0B09
	s_cselect_b32 s9, s10, s9                                  // 00000000176C: 9809090A
	s_cselect_b32 s7, s7, s10                                  // 000000001770: 98070A07
	s_add_i32 s8, s8, s6                                       // 000000001774: 81080608
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001778: BF8700C9
	s_sub_i32 s6, 32, s8                                       // 00000000177C: 818608A0
	s_cmp_eq_u32 s8, 0                                         // 000000001780: BF068008
	v_mov_b32_e32 v0, s6                                       // 000000001784: 7E000206
	s_cselect_b32 s8, -1, 0                                    // 000000001788: 980880C1
	v_alignbit_b32 v1, s3, s11, v0                             // 00000000178C: D6160001 04001603
	v_alignbit_b32 v2, s11, s9, v0                             // 000000001794: D6160002 0400120B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000179C: BF870112
	v_readfirstlane_b32 s6, v1                                 // 0000000017A0: 7E0C0501
	v_cndmask_b32_e64 v1, v2, s11, s8                          // 0000000017A4: D5010001 00201702
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017AC: BF870002
	s_cselect_b32 s3, s3, s6                                   // 0000000017B0: 98030603
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017B4: BF870481
	v_alignbit_b32 v2, s3, v1, 30                              // 0000000017B8: D6160002 027A0203
	s_bfe_u32 s10, s3, 0x1001d                                 // 0000000017C0: 930AFF03 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017C8: BF870009
	s_sub_i32 s6, 0, s10                                       // 0000000017CC: 81860A80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 0000000017D0: BF870481
	v_xor_b32_e32 v2, s6, v2                                   // 0000000017D4: 3A040406
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870091
	v_clz_i32_u32_e32 v3, v2                                   // 0000000017DC: 7E067302
	v_min_u32_e32 v3, 32, v3                                   // 0000000017E0: 260606A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000017E4: BF870131
	v_lshlrev_b32_e32 v5, 23, v3                               // 0000000017E8: 300A0697
	v_alignbit_b32 v0, s9, s7, v0                              // 0000000017EC: D6160000 04000E09
	v_sub_nc_u32_e32 v4, 31, v3                                // 0000000017F4: 4C08069F
	v_cndmask_b32_e64 v0, v0, s9, s8                           // 0000000017F8: D5010000 00201300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001800: BF870121
	v_alignbit_b32 v1, v1, v0, 30                              // 000000001804: D6160001 027A0101
	v_alignbit_b32 v0, v0, s7, 30                              // 00000000180C: D6160000 02780F00
	v_xor_b32_e32 v1, s6, v1                                   // 000000001814: 3A020206
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001818: BF870002
	v_xor_b32_e32 v0, s6, v0                                   // 00000000181C: 3A000006
	s_lshr_b32 s6, s3, 29                                      // 000000001820: 85069D03
	s_lshr_b32 s3, s3, 30                                      // 000000001824: 85039E03
	s_lshl_b32 s6, s6, 31                                      // 000000001828: 84069F06
	v_alignbit_b32 v2, v2, v1, v4                              // 00000000182C: D6160002 04120302
	v_alignbit_b32 v0, v1, v0, v4                              // 000000001834: D6160000 04120101
	s_or_b32 s7, s6, 0.5                                       // 00000000183C: 8C07F006
	s_add_i32 s3, s10, s3                                      // 000000001840: 8103030A
	v_sub_nc_u32_e32 v5, s7, v5                                // 000000001844: 4C0A0A07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001848: BF870092
	v_alignbit_b32 v1, v2, v0, 9                               // 00000000184C: D6160001 02260102
	v_clz_i32_u32_e32 v4, v1                                   // 000000001854: 7E087301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001858: BF870091
	v_min_u32_e32 v4, 32, v4                                   // 00000000185C: 260808A0
	v_sub_nc_u32_e32 v6, 31, v4                                // 000000001860: 4C0C089F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001864: BF870121
	v_alignbit_b32 v0, v1, v0, v6                              // 000000001868: D6160000 041A0101
	v_lshrrev_b32_e32 v1, 9, v2                                // 000000001870: 32020489
	v_lshrrev_b32_e32 v0, 9, v0                                // 000000001874: 32000089
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001878: BF8700A2
	v_or_b32_e32 v1, v1, v5                                    // 00000000187C: 38020B01
	v_add_nc_u32_e32 v3, v4, v3                                // 000000001880: 4A060704
	v_lshlrev_b32_e32 v2, 23, v3                               // 000000001884: 30040697
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001888: BF870211
	v_sub_nc_u32_e32 v0, v0, v2                                // 00000000188C: 4C000500
	v_mul_f32_e32 v2, 0x3fc90fda, v1                           // 000000001890: 100402FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001898: BF870112
	v_add_nc_u32_e32 v0, 0x33000000, v0                        // 00000000189C: 4A0000FF 33000000
	v_fma_f32 v3, 0x3fc90fda, v1, -v2                          // 0000000018A4: D6130003 840A02FF 3FC90FDA
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000018B0: BF870112
	v_or_b32_e32 v0, s6, v0                                    // 0000000018B4: 38000006
	v_fmac_f32_e32 v3, 0x33a22168, v1                          // 0000000018B8: 560602FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018C0: BF870091
	v_fmac_f32_e32 v3, 0x3fc90fda, v0                          // 0000000018C4: 560600FF 3FC90FDA
	v_add_f32_e32 v0, v2, v3                                   // 0000000018CC: 06000702
	s_cbranch_execz 2                                          // 0000000018D0: BFA50002 <E_3n78+0x2dc>
	v_mov_b32_e32 v1, s3                                       // 0000000018D4: 7E020203
	s_branch 15                                                // 0000000018D8: BFA0000F <E_3n78+0x318>
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018DC: BF870091
	v_mul_f32_e64 v0, 0x3f22f983, s2                           // 0000000018E0: D5080000 000004FF 3F22F983
	v_rndne_f32_e32 v1, v0                                     // 0000000018EC: 7E024700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F0: BF870091
	v_fma_f32 v0, 0xbfc90fda, v1, s2                           // 0000000018F4: D6130000 000A02FF BFC90FDA
	v_fmac_f32_e32 v0, 0xb3a22168, v1                          // 000000001900: 560002FF B3A22168
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001908: BF870001
	v_fmac_f32_e32 v0, 0xa7c234c4, v1                          // 00000000190C: 560002FF A7C234C4
	v_cvt_i32_f32_e32 v1, v1                                   // 000000001914: 7E021101
	v_sub_f32_e64 v2, 0x3fc90fdb, s2                           // 000000001918: D5040002 000004FF 3FC90FDB
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001924: BF870121
	v_cmp_ngt_f32_e64 s3, 0x48000000, |v2|                     // 000000001928: D41B0203 000204FF 48000000
	v_readfirstlane_b32 s6, v2                                 // 000000001934: 7E0C0502
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001938: 8B6A037E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000193C: BF870001
	s_and_b32 s3, s6, 0x7fffffff                               // 000000001940: 8B03FF06 7FFFFFFF
	s_cbranch_vccz 164                                         // 000000001948: BFA300A4 <E_3n78+0x5dc>
	s_and_b32 s6, s3, 0x7fffff                                 // 00000000194C: 8B06FF03 007FFFFF
	s_lshr_b32 s7, s3, 23                                      // 000000001954: 85079703
	s_bitset1_b32 s6, 23                                       // 000000001958: BE861297
	s_addk_i32 s7, 0xff88                                      // 00000000195C: B787FF88
	s_mul_hi_u32 s8, s6, 0xfe5163ab                            // 000000001960: 9688FF06 FE5163AB
	s_mul_i32 s9, s6, 0x3c439041                               // 000000001968: 9609FF06 3C439041
	s_mul_hi_u32 s10, s6, 0x3c439041                           // 000000001970: 968AFF06 3C439041
	s_add_u32 s8, s8, s9                                       // 000000001978: 80080908
	s_addc_u32 s9, 0, s10                                      // 00000000197C: 82090A80
	s_mul_i32 s10, s6, 0xdb629599                              // 000000001980: 960AFF06 DB629599
	s_mul_hi_u32 s11, s6, 0xdb629599                           // 000000001988: 968BFF06 DB629599
	s_add_u32 s9, s9, s10                                      // 000000001990: 80090A09
	s_addc_u32 s10, 0, s11                                     // 000000001994: 820A0B80
	s_mul_i32 s11, s6, 0xf534ddc0                              // 000000001998: 960BFF06 F534DDC0
	s_mul_hi_u32 s12, s6, 0xf534ddc0                           // 0000000019A0: 968CFF06 F534DDC0
	s_add_u32 s10, s10, s11                                    // 0000000019A8: 800A0B0A
	s_addc_u32 s11, 0, s12                                     // 0000000019AC: 820B0C80
	s_mul_i32 s12, s6, 0xfc2757d1                              // 0000000019B0: 960CFF06 FC2757D1
	s_mul_hi_u32 s13, s6, 0xfc2757d1                           // 0000000019B8: 968DFF06 FC2757D1
	s_add_u32 s11, s11, s12                                    // 0000000019C0: 800B0C0B
	s_addc_u32 s12, 0, s13                                     // 0000000019C4: 820C0D80
	s_mul_i32 s13, s6, 0x4e441529                              // 0000000019C8: 960DFF06 4E441529
	s_mul_hi_u32 s14, s6, 0x4e441529                           // 0000000019D0: 968EFF06 4E441529
	s_add_u32 s12, s12, s13                                    // 0000000019D8: 800C0D0C
	s_addc_u32 s13, 0, s14                                     // 0000000019DC: 820D0E80
	s_cmp_gt_u32 s7, 63                                        // 0000000019E0: BF08BF07
	s_mul_i32 s14, s6, 0xfe5163ab                              // 0000000019E4: 960EFF06 FE5163AB
	s_mul_hi_u32 s15, s6, 0xa2f9836e                           // 0000000019EC: 968FFF06 A2F9836E
	s_mul_i32 s6, s6, 0xa2f9836e                               // 0000000019F4: 9606FF06 A2F9836E
	s_cselect_b32 s16, s9, s11                                 // 0000000019FC: 98100B09
	s_cselect_b32 s8, s8, s10                                  // 000000001A00: 98080A08
	s_cselect_b32 s9, s14, s9                                  // 000000001A04: 9809090E
	s_add_u32 s6, s13, s6                                      // 000000001A08: 8006060D
	s_addc_u32 s13, 0, s15                                     // 000000001A0C: 820D0F80
	s_cmp_gt_u32 s7, 63                                        // 000000001A10: BF08BF07
	s_cselect_b32 s14, 0xffffffc0, 0                           // 000000001A14: 980E80FF FFFFFFC0
	s_cselect_b32 s10, s10, s12                                // 000000001A1C: 980A0C0A
	s_cselect_b32 s6, s11, s6                                  // 000000001A20: 9806060B
	s_cselect_b32 s11, s12, s13                                // 000000001A24: 980B0D0C
	s_add_i32 s14, s14, s7                                     // 000000001A28: 810E070E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A2C: BF870009
	s_cmp_gt_u32 s14, 31                                       // 000000001A30: BF089F0E
	s_cselect_b32 s7, 0xffffffe0, 0                            // 000000001A34: 980780FF FFFFFFE0
	s_cselect_b32 s12, s10, s6                                 // 000000001A3C: 980C060A
	s_cselect_b32 s6, s6, s11                                  // 000000001A40: 98060B06
	s_cselect_b32 s10, s16, s10                                // 000000001A44: 980A0A10
	s_cselect_b32 s11, s8, s16                                 // 000000001A48: 980B1008
	s_cselect_b32 s8, s9, s8                                   // 000000001A4C: 98080809
	s_add_i32 s7, s7, s14                                      // 000000001A50: 81070E07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A54: BF870009
	s_cmp_gt_u32 s7, 31                                        // 000000001A58: BF089F07
	s_cselect_b32 s9, 0xffffffe0, 0                            // 000000001A5C: 980980FF FFFFFFE0
	s_cselect_b32 s6, s12, s6                                  // 000000001A64: 9806060C
	s_cselect_b32 s12, s10, s12                                // 000000001A68: 980C0C0A
	s_cselect_b32 s10, s11, s10                                // 000000001A6C: 980A0A0B
	s_cselect_b32 s8, s8, s11                                  // 000000001A70: 98080B08
	s_add_i32 s9, s9, s7                                       // 000000001A74: 81090709
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001A78: BF8700C9
	s_sub_i32 s7, 32, s9                                       // 000000001A7C: 818709A0
	s_cmp_eq_u32 s9, 0                                         // 000000001A80: BF068009
	v_mov_b32_e32 v3, s7                                       // 000000001A84: 7E060207
	s_cselect_b32 s9, -1, 0                                    // 000000001A88: 980980C1
	v_alignbit_b32 v4, s6, s12, v3                             // 000000001A8C: D6160004 040C1806
	v_alignbit_b32 v5, s12, s10, v3                            // 000000001A94: D6160005 040C140C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001A9C: BF870112
	v_readfirstlane_b32 s7, v4                                 // 000000001AA0: 7E0E0504
	v_cndmask_b32_e64 v4, v5, s12, s9                          // 000000001AA4: D5010004 00241905
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001AAC: BF870002
	s_cselect_b32 s6, s6, s7                                   // 000000001AB0: 98060706
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001AB4: BF870481
	v_alignbit_b32 v5, s6, v4, 30                              // 000000001AB8: D6160005 027A0806
	s_bfe_u32 s11, s6, 0x1001d                                 // 000000001AC0: 930BFF06 0001001D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001AC8: BF870009
	s_sub_i32 s7, 0, s11                                       // 000000001ACC: 81870B80
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001AD0: BF870481
	v_xor_b32_e32 v5, s7, v5                                   // 000000001AD4: 3A0A0A07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AD8: BF870091
	v_clz_i32_u32_e32 v6, v5                                   // 000000001ADC: 7E0C7305
	v_min_u32_e32 v6, 32, v6                                   // 000000001AE0: 260C0CA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001AE4: BF870131
	v_lshlrev_b32_e32 v8, 23, v6                               // 000000001AE8: 30100C97
	v_alignbit_b32 v3, s10, s8, v3                             // 000000001AEC: D6160003 040C100A
	v_sub_nc_u32_e32 v7, 31, v6                                // 000000001AF4: 4C0E0C9F
	v_cndmask_b32_e64 v3, v3, s10, s9                          // 000000001AF8: D5010003 00241503
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001B00: BF870121
	v_alignbit_b32 v4, v4, v3, 30                              // 000000001B04: D6160004 027A0704
	v_alignbit_b32 v3, v3, s8, 30                              // 000000001B0C: D6160003 02781103
	v_xor_b32_e32 v4, s7, v4                                   // 000000001B14: 3A080807
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B18: BF870002
	v_xor_b32_e32 v3, s7, v3                                   // 000000001B1C: 3A060607
	s_lshr_b32 s7, s6, 29                                      // 000000001B20: 85079D06
	s_lshr_b32 s6, s6, 30                                      // 000000001B24: 85069E06
	s_lshl_b32 s7, s7, 31                                      // 000000001B28: 84079F07
	v_alignbit_b32 v5, v5, v4, v7                              // 000000001B2C: D6160005 041E0905
	v_alignbit_b32 v3, v4, v3, v7                              // 000000001B34: D6160003 041E0704
	s_or_b32 s8, s7, 0.5                                       // 000000001B3C: 8C08F007
	s_add_i32 s6, s11, s6                                      // 000000001B40: 8106060B
	v_sub_nc_u32_e32 v8, s8, v8                                // 000000001B44: 4C101008
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
	v_or_b32_e32 v3, s7, v3                                    // 000000001BB4: 38060607
	v_fmac_f32_e32 v6, 0x33a22168, v4                          // 000000001BB8: 560C08FF 33A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC0: BF870091
	v_fmac_f32_e32 v6, 0x3fc90fda, v3                          // 000000001BC4: 560C06FF 3FC90FDA
	v_add_f32_e32 v3, v5, v6                                   // 000000001BCC: 06060D05
	s_cbranch_execz 2                                          // 000000001BD0: BFA50002 <E_3n78+0x5dc>
	v_mov_b32_e32 v4, s6                                       // 000000001BD4: 7E080206
	s_branch 14                                                // 000000001BD8: BFA0000E <E_3n78+0x614>
	v_mul_f32_e64 v3, 0x3f22f983, s3                           // 000000001BDC: D5080003 000006FF 3F22F983
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BE8: BF870091
	v_rndne_f32_e32 v4, v3                                     // 000000001BEC: 7E084703
	v_fma_f32 v3, 0xbfc90fda, v4, s3                           // 000000001BF0: D6130003 000E08FF BFC90FDA
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
	s_xor_b32 s6, s2, s2                                       // 000000001C40: 8D060202
	v_fmaak_f32 v10, s7, v7, 0x3c0881c4                        // 000000001C44: 5A140E07 3C0881C4
	v_xor_b32_e32 v2, s3, v2                                   // 000000001C4C: 3A040403
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
	v_xor_b32_e32 v6, s6, v6                                   // 000000001D00: 3A0C0C06
	v_cmp_class_f32_e64 vcc_lo, s2, 0x1f8                      // 000000001D04: D47E006A 0001FE02 000001F8
	v_xor_b32_e32 v1, v2, v1                                   // 000000001D10: 3A020302
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D14: BF870093
	v_xor_b32_e32 v0, v6, v0                                   // 000000001D18: 3A000106
	v_cndmask_b32_e32 v0, 0x7fc00000, v0, vcc_lo               // 000000001D1C: 020000FF 7FC00000
	v_cmp_class_f32_e64 vcc_lo, s3, 0x1f8                      // 000000001D24: D47E006A 0001FE03 000001F8
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
