
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_10_10_10>:
	s_mul_i32 s2, s15, 10                                      // 000000001600: 96028A0F
	s_load_b64 s[0:1], s[0:1], null                            // 000000001604: F4040000 F8000000
	s_mul_hi_i32 s3, s2, 0x2e8ba2e9                            // 00000000160C: 9703FF02 2E8BA2E9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001614: BF8704A9
	s_lshr_b32 s4, s3, 31                                      // 000000001618: 85049F03
	s_ashr_i32 s3, s3, 1                                       // 00000000161C: 86038103
	s_add_i32 s3, s3, s4                                       // 000000001620: 81030403
	s_add_i32 s4, s14, 2                                       // 000000001624: 8104820E
	s_mul_i32 s3, s3, 11                                       // 000000001628: 96038B03
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000162C: BF870499
	s_sub_i32 s3, s2, s3                                       // 000000001630: 81830302
	s_cmp_lt_i32 s3, 1                                         // 000000001634: BF048103
	s_mul_hi_i32 s3, s4, 0x2e8ba2e9                            // 000000001638: 9703FF04 2E8BA2E9
	s_cselect_b32 s5, -1, 0                                    // 000000001640: 980580C1
	s_lshr_b32 s6, s3, 31                                      // 000000001644: 85069F03
	s_ashr_i32 s3, s3, 1                                       // 000000001648: 86038103
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000164C: BF870499
	s_add_i32 s3, s3, s6                                       // 000000001650: 81030603
	s_mul_i32 s3, s3, 11                                       // 000000001654: 96038B03
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001658: BF870499
	s_sub_i32 s3, s4, s3                                       // 00000000165C: 81830304
	s_cmp_lt_i32 s3, 1                                         // 000000001660: BF048103
	s_cselect_b32 s3, -1, 0                                    // 000000001664: 980380C1
	s_or_b32 s4, s2, 1                                         // 000000001668: 8C048102
	s_and_b32 s3, s5, s3                                       // 00000000166C: 8B030305
	s_mul_hi_i32 s6, s4, 0x2e8ba2e9                            // 000000001670: 9706FF04 2E8BA2E9
	v_cndmask_b32_e64 v0, 0, 1.0, s3                           // 000000001678: D5010000 000DE480
	s_lshr_b32 s7, s6, 31                                      // 000000001680: 85079F06
	s_ashr_i32 s6, s6, 1                                       // 000000001684: 86068106
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001688: BF870499
	s_add_i32 s6, s6, s7                                       // 00000000168C: 81060706
	s_mul_i32 s6, s6, 11                                       // 000000001690: 96068B06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001694: BF870499
	s_sub_i32 s4, s4, s6                                       // 000000001698: 81840604
	s_cmp_lt_i32 s4, 1                                         // 00000000169C: BF048104
	s_cselect_b32 s4, -1, 0                                    // 0000000016A0: 980480C1
	s_add_i32 s5, s14, 3                                       // 0000000016A4: 8105830E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000016A8: BF870499
	s_mul_hi_i32 s6, s5, 0x2e8ba2e9                            // 0000000016AC: 9706FF05 2E8BA2E9
	s_lshr_b32 s7, s6, 31                                      // 0000000016B4: 85079F06
	s_ashr_i32 s6, s6, 1                                       // 0000000016B8: 86068106
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000016BC: BF870499
	s_add_i32 s6, s6, s7                                       // 0000000016C0: 81060706
	s_mul_i32 s6, s6, 11                                       // 0000000016C4: 96068B06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000016C8: BF870499
	s_sub_i32 s5, s5, s6                                       // 0000000016CC: 81850605
	s_cmp_lt_i32 s5, 1                                         // 0000000016D0: BF048105
	s_cselect_b32 s5, -1, 0                                    // 0000000016D4: 980580C1
	s_add_i32 s6, s2, 2                                        // 0000000016D8: 81068202
	s_and_b32 s4, s4, s5                                       // 0000000016DC: 8B040504
	s_mul_hi_i32 s7, s6, 0x2e8ba2e9                            // 0000000016E0: 9707FF06 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 0000000016E8: D5010001 0011E480
	s_lshr_b32 s8, s7, 31                                      // 0000000016F0: 85089F07
	s_ashr_i32 s7, s7, 1                                       // 0000000016F4: 86078107
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016F8: BF870099
	s_add_i32 s7, s7, s8                                       // 0000000016FC: 81070807
	v_add_f32_e32 v0, v0, v1                                   // 000000001700: 06000300
	s_mul_i32 s7, s7, 11                                       // 000000001704: 96078B07
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001708: BF870499
	s_sub_i32 s6, s6, s7                                       // 00000000170C: 81860706
	s_cmp_lt_i32 s6, 1                                         // 000000001710: BF048106
	s_cselect_b32 s5, -1, 0                                    // 000000001714: 980580C1
	s_add_i32 s6, s14, 4                                       // 000000001718: 8106840E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000171C: BF870499
	s_mul_hi_i32 s7, s6, 0x2e8ba2e9                            // 000000001720: 9707FF06 2E8BA2E9
	s_lshr_b32 s8, s7, 31                                      // 000000001728: 85089F07
	s_ashr_i32 s7, s7, 1                                       // 00000000172C: 86078107
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001730: BF870499
	s_add_i32 s7, s7, s8                                       // 000000001734: 81070807
	s_mul_i32 s7, s7, 11                                       // 000000001738: 96078B07
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_sub_i32 s6, s6, s7                                       // 000000001740: 81860706
	s_cmp_lt_i32 s6, 1                                         // 000000001744: BF048106
	s_cselect_b32 s6, -1, 0                                    // 000000001748: 980680C1
	s_add_i32 s7, s2, 3                                        // 00000000174C: 81078302
	s_and_b32 s5, s5, s6                                       // 000000001750: 8B050605
	s_mul_hi_i32 s8, s7, 0x2e8ba2e9                            // 000000001754: 9708FF07 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s5                           // 00000000175C: D5010001 0015E480
	s_lshr_b32 s9, s8, 31                                      // 000000001764: 85099F08
	s_ashr_i32 s8, s8, 1                                       // 000000001768: 86088108
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000176C: BF870099
	s_add_i32 s8, s8, s9                                       // 000000001770: 81080908
	v_add_f32_e32 v0, v0, v1                                   // 000000001774: 06000300
	s_mul_i32 s8, s8, 11                                       // 000000001778: 96088B08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000177C: BF870499
	s_sub_i32 s7, s7, s8                                       // 000000001780: 81870807
	s_cmp_lt_i32 s7, 1                                         // 000000001784: BF048107
	s_cselect_b32 s6, -1, 0                                    // 000000001788: 980680C1
	s_add_i32 s7, s14, 5                                       // 00000000178C: 8107850E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001790: BF870499
	s_mul_hi_i32 s8, s7, 0x2e8ba2e9                            // 000000001794: 9708FF07 2E8BA2E9
	s_lshr_b32 s9, s8, 31                                      // 00000000179C: 85099F08
	s_ashr_i32 s8, s8, 1                                       // 0000000017A0: 86088108
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017A4: BF870499
	s_add_i32 s8, s8, s9                                       // 0000000017A8: 81080908
	s_mul_i32 s8, s8, 11                                       // 0000000017AC: 96088B08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017B0: BF870499
	s_sub_i32 s7, s7, s8                                       // 0000000017B4: 81870807
	s_cmp_lt_i32 s7, 1                                         // 0000000017B8: BF048107
	s_cselect_b32 s7, -1, 0                                    // 0000000017BC: 980780C1
	s_add_i32 s8, s2, 4                                        // 0000000017C0: 81088402
	s_and_b32 s6, s6, s7                                       // 0000000017C4: 8B060706
	s_mul_hi_i32 s9, s8, 0x2e8ba2e9                            // 0000000017C8: 9709FF08 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s6                           // 0000000017D0: D5010001 0019E480
	s_lshr_b32 s10, s9, 31                                     // 0000000017D8: 850A9F09
	s_ashr_i32 s9, s9, 1                                       // 0000000017DC: 86098109
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E0: BF870099
	s_add_i32 s9, s9, s10                                      // 0000000017E4: 81090A09
	v_add_f32_e32 v0, v0, v1                                   // 0000000017E8: 06000300
	s_mul_i32 s9, s9, 11                                       // 0000000017EC: 96098B09
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017F0: BF870499
	s_sub_i32 s8, s8, s9                                       // 0000000017F4: 81880908
	s_cmp_lt_i32 s8, 1                                         // 0000000017F8: BF048108
	s_cselect_b32 s7, -1, 0                                    // 0000000017FC: 980780C1
	s_add_i32 s8, s14, 6                                       // 000000001800: 8108860E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001804: BF870499
	s_mul_hi_i32 s9, s8, 0x2e8ba2e9                            // 000000001808: 9709FF08 2E8BA2E9
	s_lshr_b32 s10, s9, 31                                     // 000000001810: 850A9F09
	s_ashr_i32 s9, s9, 1                                       // 000000001814: 86098109
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001818: BF870499
	s_add_i32 s9, s9, s10                                      // 00000000181C: 81090A09
	s_mul_i32 s9, s9, 11                                       // 000000001820: 96098B09
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001824: BF870499
	s_sub_i32 s8, s8, s9                                       // 000000001828: 81880908
	s_cmp_lt_i32 s8, 1                                         // 00000000182C: BF048108
	s_cselect_b32 s8, -1, 0                                    // 000000001830: 980880C1
	s_add_i32 s9, s2, 5                                        // 000000001834: 81098502
	s_and_b32 s7, s7, s8                                       // 000000001838: 8B070807
	s_mul_hi_i32 s10, s9, 0x2e8ba2e9                           // 00000000183C: 970AFF09 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s7                           // 000000001844: D5010001 001DE480
	s_lshr_b32 s11, s10, 31                                    // 00000000184C: 850B9F0A
	s_ashr_i32 s10, s10, 1                                     // 000000001850: 860A810A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001854: BF870099
	s_add_i32 s10, s10, s11                                    // 000000001858: 810A0B0A
	v_add_f32_e32 v0, v0, v1                                   // 00000000185C: 06000300
	s_mul_i32 s10, s10, 11                                     // 000000001860: 960A8B0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001864: BF870499
	s_sub_i32 s9, s9, s10                                      // 000000001868: 81890A09
	s_cmp_lt_i32 s9, 1                                         // 00000000186C: BF048109
	s_cselect_b32 s8, -1, 0                                    // 000000001870: 980880C1
	s_add_i32 s9, s14, 7                                       // 000000001874: 8109870E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001878: BF870499
	s_mul_hi_i32 s10, s9, 0x2e8ba2e9                           // 00000000187C: 970AFF09 2E8BA2E9
	s_lshr_b32 s11, s10, 31                                    // 000000001884: 850B9F0A
	s_ashr_i32 s10, s10, 1                                     // 000000001888: 860A810A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000188C: BF870499
	s_add_i32 s10, s10, s11                                    // 000000001890: 810A0B0A
	s_mul_i32 s10, s10, 11                                     // 000000001894: 960A8B0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001898: BF870499
	s_sub_i32 s9, s9, s10                                      // 00000000189C: 81890A09
	s_cmp_lt_i32 s9, 1                                         // 0000000018A0: BF048109
	s_cselect_b32 s9, -1, 0                                    // 0000000018A4: 980980C1
	s_add_i32 s10, s2, 6                                       // 0000000018A8: 810A8602
	s_and_b32 s8, s8, s9                                       // 0000000018AC: 8B080908
	s_mul_hi_i32 s11, s10, 0x2e8ba2e9                          // 0000000018B0: 970BFF0A 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s8                           // 0000000018B8: D5010001 0021E480
	s_lshr_b32 s12, s11, 31                                    // 0000000018C0: 850C9F0B
	s_ashr_i32 s11, s11, 1                                     // 0000000018C4: 860B810B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018C8: BF870099
	s_add_i32 s11, s11, s12                                    // 0000000018CC: 810B0C0B
	v_add_f32_e32 v0, v0, v1                                   // 0000000018D0: 06000300
	s_mul_i32 s11, s11, 11                                     // 0000000018D4: 960B8B0B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018D8: BF870499
	s_sub_i32 s10, s10, s11                                    // 0000000018DC: 818A0B0A
	s_cmp_lt_i32 s10, 1                                        // 0000000018E0: BF04810A
	s_cselect_b32 s9, -1, 0                                    // 0000000018E4: 980980C1
	s_add_i32 s10, s14, 8                                      // 0000000018E8: 810A880E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018EC: BF870499
	s_mul_hi_i32 s11, s10, 0x2e8ba2e9                          // 0000000018F0: 970BFF0A 2E8BA2E9
	s_lshr_b32 s12, s11, 31                                    // 0000000018F8: 850C9F0B
	s_ashr_i32 s11, s11, 1                                     // 0000000018FC: 860B810B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001900: BF870499
	s_add_i32 s11, s11, s12                                    // 000000001904: 810B0C0B
	s_mul_i32 s11, s11, 11                                     // 000000001908: 960B8B0B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000190C: BF870499
	s_sub_i32 s10, s10, s11                                    // 000000001910: 818A0B0A
	s_cmp_lt_i32 s10, 1                                        // 000000001914: BF04810A
	s_cselect_b32 s10, -1, 0                                   // 000000001918: 980A80C1
	s_add_i32 s11, s2, 7                                       // 00000000191C: 810B8702
	s_and_b32 s9, s9, s10                                      // 000000001920: 8B090A09
	s_mul_hi_i32 s12, s11, 0x2e8ba2e9                          // 000000001924: 970CFF0B 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s9                           // 00000000192C: D5010001 0025E480
	s_lshr_b32 s13, s12, 31                                    // 000000001934: 850D9F0C
	s_ashr_i32 s12, s12, 1                                     // 000000001938: 860C810C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000193C: BF870099
	s_add_i32 s12, s12, s13                                    // 000000001940: 810C0D0C
	v_add_f32_e32 v0, v0, v1                                   // 000000001944: 06000300
	s_mul_i32 s12, s12, 11                                     // 000000001948: 960C8B0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000194C: BF870499
	s_sub_i32 s11, s11, s12                                    // 000000001950: 818B0C0B
	s_cmp_lt_i32 s11, 1                                        // 000000001954: BF04810B
	s_cselect_b32 s10, -1, 0                                   // 000000001958: 980A80C1
	s_add_i32 s11, s14, 9                                      // 00000000195C: 810B890E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001960: BF870499
	s_mul_hi_i32 s12, s11, 0x2e8ba2e9                          // 000000001964: 970CFF0B 2E8BA2E9
	s_lshr_b32 s13, s12, 31                                    // 00000000196C: 850D9F0C
	s_ashr_i32 s12, s12, 1                                     // 000000001970: 860C810C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001974: BF870499
	s_add_i32 s12, s12, s13                                    // 000000001978: 810C0D0C
	s_mul_i32 s12, s12, 11                                     // 00000000197C: 960C8B0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001980: BF870499
	s_sub_i32 s11, s11, s12                                    // 000000001984: 818B0C0B
	s_cmp_lt_i32 s11, 1                                        // 000000001988: BF04810B
	s_cselect_b32 s3, -1, 0                                    // 00000000198C: 980380C1
	s_add_i32 s11, s2, 8                                       // 000000001990: 810B8802
	s_and_b32 s3, s10, s3                                      // 000000001994: 8B03030A
	s_mul_hi_i32 s4, s11, 0x2e8ba2e9                           // 000000001998: 9704FF0B 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s3                           // 0000000019A0: D5010001 000DE480
	s_lshr_b32 s12, s4, 31                                     // 0000000019A8: 850C9F04
	s_ashr_i32 s4, s4, 1                                       // 0000000019AC: 86048104
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019B0: BF870099
	s_add_i32 s4, s4, s12                                      // 0000000019B4: 81040C04
	v_add_f32_e32 v0, v0, v1                                   // 0000000019B8: 06000300
	s_mul_i32 s4, s4, 11                                       // 0000000019BC: 96048B04
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019C0: BF870499
	s_sub_i32 s4, s11, s4                                      // 0000000019C4: 8184040B
	s_cmp_lt_i32 s4, 1                                         // 0000000019C8: BF048104
	s_cselect_b32 s4, -1, 0                                    // 0000000019CC: 980480C1
	s_add_i32 s5, s14, 10                                      // 0000000019D0: 81058A0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019D4: BF870499
	s_mul_hi_i32 s6, s5, 0x2e8ba2e9                            // 0000000019D8: 9706FF05 2E8BA2E9
	s_lshr_b32 s10, s6, 31                                     // 0000000019E0: 850A9F06
	s_ashr_i32 s6, s6, 1                                       // 0000000019E4: 86068106
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019E8: BF870499
	s_add_i32 s6, s6, s10                                      // 0000000019EC: 81060A06
	s_mul_i32 s6, s6, 11                                       // 0000000019F0: 96068B06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019F4: BF870499
	s_sub_i32 s5, s5, s6                                       // 0000000019F8: 81850605
	s_cmp_lt_i32 s5, 1                                         // 0000000019FC: BF048105
	s_cselect_b32 s5, -1, 0                                    // 000000001A00: 980580C1
	s_add_i32 s6, s2, 9                                        // 000000001A04: 81068902
	s_and_b32 s4, s4, s5                                       // 000000001A08: 8B040504
	s_mul_hi_i32 s7, s6, 0x2e8ba2e9                            // 000000001A0C: 9707FF06 2E8BA2E9
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 000000001A14: D5010001 0011E480
	s_lshr_b32 s8, s7, 31                                      // 000000001A1C: 85089F07
	s_ashr_i32 s7, s7, 1                                       // 000000001A20: 86078107
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A24: BF870099
	s_add_i32 s7, s7, s8                                       // 000000001A28: 81070807
	v_add_f32_e32 v0, v0, v1                                   // 000000001A2C: 06000300
	s_mul_i32 s7, s7, 11                                       // 000000001A30: 96078B07
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001A34: BF870499
	s_sub_i32 s6, s6, s7                                       // 000000001A38: 81860706
	s_cmp_lt_i32 s6, 1                                         // 000000001A3C: BF048106
	s_cselect_b32 s5, -1, 0                                    // 000000001A40: 980580C1
	s_add_i32 s6, s14, 11                                      // 000000001A44: 81068B0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001A48: BF870499
	s_mul_hi_i32 s3, s6, 0x2e8ba2e9                            // 000000001A4C: 9703FF06 2E8BA2E9
	s_lshr_b32 s7, s3, 31                                      // 000000001A54: 85079F03
	s_ashr_i32 s3, s3, 1                                       // 000000001A58: 86038103
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001A5C: BF870499
	s_add_i32 s3, s3, s7                                       // 000000001A60: 81030703
	s_mul_i32 s3, s3, 11                                       // 000000001A64: 96038B03
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001A68: BF870499
	s_sub_i32 s3, s6, s3                                       // 000000001A6C: 81830306
	s_cmp_lt_i32 s3, 1                                         // 000000001A70: BF048103
	s_cselect_b32 s4, -1, 0                                    // 000000001A74: 980480C1
	s_ashr_i32 s3, s2, 31                                      // 000000001A78: 86039F02
	s_and_b32 s4, s5, s4                                       // 000000001A7C: 8B040405
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001A80: 84828202
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 000000001A84: D5010001 0011E480
	s_waitcnt lgkmcnt(0)                                       // 000000001A8C: BF89FC07
	s_add_u32 s2, s0, s2                                       // 000000001A90: 80020200
	s_addc_u32 s3, s1, s3                                      // 000000001A94: 82030301
	s_ashr_i32 s15, s14, 31                                    // 000000001A98: 860F9F0E
	v_dual_add_f32 v0, v0, v1 :: v_dual_mov_b32 v1, 0          // 000000001A9C: C9100300 00000080
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001AA4: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001AA8: BF870009
	s_add_u32 s0, s2, s0                                       // 000000001AAC: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001AB0: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001AB4: DC6A0000 00000001
	s_nop 0                                                    // 000000001ABC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001AC0: BFB60003
	s_endpgm                                                   // 000000001AC4: BFB00000
