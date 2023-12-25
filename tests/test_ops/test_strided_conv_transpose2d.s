
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_4_9_7_4_3_3n1>:
	s_load_b64 s[4:5], s[0:1], 0x10                            // 000000001700: F4040100 F8000010
	s_mul_hi_i32 s2, s13, 0x92492493                           // 000000001708: 9702FF0D 92492493
	v_mov_b32_e32 v0, 0                                        // 000000001710: 7E000280
	s_add_i32 s2, s2, s13                                      // 000000001714: 81020D02
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001718: BF870009
	s_lshr_b32 s3, s2, 31                                      // 00000000171C: 85039F02
	s_ashr_i32 s6, s2, 2                                       // 000000001720: 86068202
	s_mul_i32 s2, s14, 9                                       // 000000001724: 9602890E
	s_add_i32 s25, s6, s3                                      // 000000001728: 81190306
	s_ashr_i32 s3, s2, 31                                      // 00000000172C: 86039F02
	s_mul_i32 s8, s25, 7                                       // 000000001730: 96088719
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001734: 84868202
	s_sub_i32 s2, s13, s8                                      // 000000001738: 8182080D
	s_waitcnt lgkmcnt(0)                                       // 00000000173C: BF89FC07
	s_add_u32 s3, s4, s6                                       // 000000001740: 80030604
	s_addc_u32 s4, s5, s7                                      // 000000001744: 82040705
	s_add_u32 s10, s3, 32                                      // 000000001748: 800AA003
	s_addc_u32 s11, s4, 0                                      // 00000000174C: 820B8004
	s_add_i32 s3, s2, 3                                        // 000000001750: 81038302
	s_add_i32 s20, s25, 1                                      // 000000001754: 81148119
	s_mul_hi_i32 s4, s3, 0x66666667                            // 000000001758: 9704FF03 66666667
	s_lshl_b32 s27, s15, 4                                     // 000000001760: 841B840F
	s_lshr_b32 s5, s4, 31                                      // 000000001764: 85059F04
	s_ashr_i32 s4, s4, 1                                       // 000000001768: 86048104
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000176C: BF870499
	s_add_i32 s26, s4, s5                                      // 000000001770: 811A0504
	s_add_i32 s9, s20, s26                                     // 000000001774: 81091A14
	s_cmp_gt_i32 s2, 1                                         // 000000001778: BF028102
	s_cselect_b32 s28, -1, 0                                   // 00000000177C: 981C80C1
	s_add_i32 s4, s13, -14                                     // 000000001780: 8104CE0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001784: BF8704C9
	s_cmp_lt_u32 s4, 49                                        // 000000001788: BF0AB104
	s_load_b128 s[4:7], s[0:1], null                           // 00000000178C: F4080100 F8000000
	s_cselect_b32 s21, -1, 0                                   // 000000001794: 981580C1
	s_lshr_b32 s0, s9, 31                                      // 000000001798: 85009F09
	s_add_i32 s0, s9, s0                                       // 00000000179C: 81000009
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A0: BF870009
	s_and_b32 s1, s0, -2                                       // 0000000017A4: 8B01C200
	s_ashr_i32 s29, s0, 1                                      // 0000000017A8: 861D8100
	s_sub_i32 s1, s9, s1                                       // 0000000017AC: 81810109
	s_mul_hi_u32 s0, s3, 0xcccccccd                            // 0000000017B0: 9680FF03 CCCCCCCD
	s_cmp_lt_i32 s1, 1                                         // 0000000017B8: BF048101
	s_mov_b32 s1, 0                                            // 0000000017BC: BE810080
	s_cselect_b32 s9, -1, 0                                    // 0000000017C0: 980980C1
	s_lshr_b32 s0, s0, 2                                       // 0000000017C4: 85008200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017C8: BF870499
	s_mul_i32 s0, s0, 5                                        // 0000000017CC: 96008500
	s_sub_i32 s0, s3, s0                                       // 0000000017D0: 81800003
	s_and_b32 s3, s21, s9                                      // 0000000017D4: 8B030915
	s_lshl_b64 s[16:17], s[0:1], 2                             // 0000000017D8: 84908200
	s_and_b32 s0, s3, s28                                      // 0000000017DC: 8B001C03
	s_waitcnt lgkmcnt(0)                                       // 0000000017E0: BF89FC07
	s_add_u32 s3, s6, s16                                      // 0000000017E4: 80031006
	s_addc_u32 s9, s7, s17                                     // 0000000017E8: 82091107
	s_add_i32 s12, s2, 4                                       // 0000000017EC: 810C8402
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017F0: BF870499
	s_mul_hi_i32 s16, s12, 0x66666667                          // 0000000017F4: 9710FF0C 66666667
	s_lshr_b32 s17, s16, 31                                    // 0000000017FC: 85119F10
	s_ashr_i32 s16, s16, 1                                     // 000000001800: 86108110
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001804: BF870009
	s_add_i32 s30, s16, s17                                    // 000000001808: 811E1110
	s_add_i32 s16, s2, -1                                      // 00000000180C: 8110C102
	s_add_i32 s17, s20, s30                                    // 000000001810: 81111E14
	s_cmp_lt_u32 s16, 5                                        // 000000001814: BF0A8510
	s_cselect_b32 s31, -1, 0                                   // 000000001818: 981F80C1
	s_lshr_b32 s16, s17, 31                                    // 00000000181C: 85109F11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001820: BF870499
	s_add_i32 s16, s17, s16                                    // 000000001824: 81101011
	s_and_b32 s18, s16, -2                                     // 000000001828: 8B12C210
	s_ashr_i32 s33, s16, 1                                     // 00000000182C: 86218110
	s_sub_i32 s17, s17, s18                                    // 000000001830: 81911211
	s_mul_i32 s16, s30, 5                                      // 000000001834: 9610851E
	s_cmp_lt_i32 s17, 1                                        // 000000001838: BF048111
	s_cselect_b32 s18, -1, 0                                   // 00000000183C: 981280C1
	s_sub_i32 s16, s12, s16                                    // 000000001840: 8190100C
	s_and_b32 s12, s21, s18                                    // 000000001844: 8B0C1215
	s_ashr_i32 s17, s16, 31                                    // 000000001848: 86119F10
	s_and_b32 s12, s12, s31                                    // 00000000184C: 8B0C1F0C
	s_lshl_b64 s[16:17], s[16:17], 2                           // 000000001850: 84908210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001854: BF8704B9
	s_add_u32 s18, s6, s16                                     // 000000001858: 80121006
	s_addc_u32 s19, s7, s17                                    // 00000000185C: 82131107
	s_add_i32 s22, s2, 5                                       // 000000001860: 81168502
	s_mul_hi_i32 s23, s22, 0x66666667                          // 000000001864: 9717FF16 66666667
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000186C: BF8704A9
	s_lshr_b32 s24, s23, 31                                    // 000000001870: 85189F17
	s_ashr_i32 s23, s23, 1                                     // 000000001874: 86178117
	s_add_i32 s34, s23, s24                                    // 000000001878: 81221817
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 00000000187C: BF8704C9
	s_add_i32 s20, s20, s34                                    // 000000001880: 81142214
	s_cmp_lt_u32 s2, 5                                         // 000000001884: BF0A8502
	s_cselect_b32 s35, -1, 0                                   // 000000001888: 982380C1
	s_lshr_b32 s23, s20, 31                                    // 00000000188C: 85179F14
	s_add_i32 s23, s20, s23                                    // 000000001890: 81171714
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001894: BF8704B9
	s_and_b32 s24, s23, -2                                     // 000000001898: 8B18C217
	s_ashr_i32 s36, s23, 1                                     // 00000000189C: 86248117
	s_sub_i32 s20, s20, s24                                    // 0000000018A0: 81941814
	s_cmp_lt_i32 s20, 1                                        // 0000000018A4: BF048114
	s_cselect_b32 s20, -1, 0                                   // 0000000018A8: 981480C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018AC: BF870499
	s_and_b32 s20, s21, s20                                    // 0000000018B0: 8B141415
	s_and_b32 s20, s20, s35                                    // 0000000018B4: 8B142314
	s_cmp_gt_u32 s2, -6                                        // 0000000018B8: BF08C602
	s_cselect_b32 s22, s22, s2                                 // 0000000018BC: 98160216
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018C0: BF870499
	s_ashr_i32 s23, s22, 31                                    // 0000000018C4: 86179F16
	s_lshl_b64 s[22:23], s[22:23], 2                           // 0000000018C8: 84968216
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018CC: BF870009
	s_add_u32 s21, s6, s22                                     // 0000000018D0: 80151606
	s_addc_u32 s22, s7, s23                                    // 0000000018D4: 82161707
	s_add_i32 s24, s25, 2                                      // 0000000018D8: 81188219
	s_add_i32 s23, s13, -7                                     // 0000000018DC: 8117C70D
	s_add_i32 s37, s24, s26                                    // 0000000018E0: 81251A18
	s_cmp_lt_u32 s23, 49                                       // 0000000018E4: BF0AB117
	s_cselect_b32 s38, -1, 0                                   // 0000000018E8: 982680C1
	s_lshr_b32 s23, s37, 31                                    // 0000000018EC: 85179F25
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018F0: BF870499
	s_add_i32 s23, s37, s23                                    // 0000000018F4: 81171725
	s_and_b32 s39, s23, -2                                     // 0000000018F8: 8B27C217
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018FC: BF870009
	s_sub_i32 s37, s37, s39                                    // 000000001900: 81A52725
	s_ashr_i32 s39, s23, 1                                     // 000000001904: 86278117
	s_cmp_lt_i32 s37, 1                                        // 000000001908: BF048125
	s_cselect_b32 s23, -1, 0                                   // 00000000190C: 981780C1
	s_add_i32 s37, s24, s30                                    // 000000001910: 81251E18
	s_and_b32 s23, s38, s23                                    // 000000001914: 8B171726
	s_lshr_b32 s40, s37, 31                                    // 000000001918: 85289F25
	s_and_b32 s23, s23, s28                                    // 00000000191C: 8B171C17
	s_add_i32 s40, s37, s40                                    // 000000001920: 81282825
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001924: BF8704B9
	s_and_b32 s41, s40, -2                                     // 000000001928: 8B29C228
	s_ashr_i32 s40, s40, 1                                     // 00000000192C: 86288128
	s_sub_i32 s37, s37, s41                                    // 000000001930: 81A52925
	s_cmp_lt_i32 s37, 1                                        // 000000001934: BF048125
	s_cselect_b32 s37, -1, 0                                   // 000000001938: 982580C1
	s_add_i32 s41, s24, s34                                    // 00000000193C: 81292218
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001940: BF870499
	s_lshr_b32 s24, s41, 31                                    // 000000001944: 85189F29
	s_add_i32 s42, s41, s24                                    // 000000001948: 812A1829
	s_and_b32 s24, s38, s37                                    // 00000000194C: 8B182526
	s_and_b32 s37, s42, -2                                     // 000000001950: 8B25C22A
	s_and_b32 s24, s24, s31                                    // 000000001954: 8B181F18
	s_sub_i32 s37, s41, s37                                    // 000000001958: 81A52529
	s_ashr_i32 s41, s42, 1                                     // 00000000195C: 8629812A
	s_cmp_lt_i32 s37, 1                                        // 000000001960: BF048125
	s_cselect_b32 s37, -1, 0                                   // 000000001964: 982580C1
	s_add_i32 s42, s25, 3                                      // 000000001968: 812A8319
	s_add_i32 s25, s13, 6                                      // 00000000196C: 8119860D
	s_and_b32 s13, s38, s37                                    // 000000001970: 8B0D2526
	s_add_i32 s26, s42, s26                                    // 000000001974: 811A1A2A
	s_and_b32 s13, s13, s35                                    // 000000001978: 8B0D230D
	s_cmp_lt_u32 s25, 55                                       // 00000000197C: BF0AB719
	s_cselect_b32 s37, -1, 0                                   // 000000001980: 982580C1
	s_lshr_b32 s25, s26, 31                                    // 000000001984: 85199F1A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001988: BF870499
	s_add_i32 s25, s26, s25                                    // 00000000198C: 8119191A
	s_and_b32 s38, s25, -2                                     // 000000001990: 8B26C219
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001994: BF870009
	s_sub_i32 s26, s26, s38                                    // 000000001998: 819A261A
	s_ashr_i32 s38, s25, 1                                     // 00000000199C: 86268119
	s_cmp_lt_i32 s26, 1                                        // 0000000019A0: BF04811A
	s_cselect_b32 s25, -1, 0                                   // 0000000019A4: 981980C1
	s_add_i32 s26, s42, s30                                    // 0000000019A8: 811A1E2A
	s_and_b32 s25, s37, s25                                    // 0000000019AC: 8B191925
	s_lshr_b32 s30, s26, 31                                    // 0000000019B0: 851E9F1A
	s_and_b32 s25, s25, s28                                    // 0000000019B4: 8B191C19
	s_add_i32 s30, s26, s30                                    // 0000000019B8: 811E1E1A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000019BC: BF8704B9
	s_and_b32 s43, s30, -2                                     // 0000000019C0: 8B2BC21E
	s_ashr_i32 s28, s30, 1                                     // 0000000019C4: 861C811E
	s_sub_i32 s26, s26, s43                                    // 0000000019C8: 819A2B1A
	s_cmp_lt_i32 s26, 1                                        // 0000000019CC: BF04811A
	s_cselect_b32 s26, -1, 0                                   // 0000000019D0: 981A80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019D4: BF870499
	s_and_b32 s26, s37, s26                                    // 0000000019D8: 8B1A1A25
	s_and_b32 s26, s26, s31                                    // 0000000019DC: 8B1A1F1A
	s_add_u32 s6, s6, s16                                      // 0000000019E0: 80061006
	s_addc_u32 s7, s7, s17                                     // 0000000019E4: 82071107
	s_add_i32 s16, s42, s34                                    // 0000000019E8: 8110222A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019EC: BF870499
	s_lshr_b32 s17, s16, 31                                    // 0000000019F0: 85119F10
	s_add_i32 s17, s16, s17                                    // 0000000019F4: 81111110
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000019F8: BF8704B9
	s_and_b32 s30, s17, -2                                     // 0000000019FC: 8B1EC211
	s_ashr_i32 s17, s17, 1                                     // 000000001A00: 86118111
	s_sub_i32 s16, s16, s30                                    // 000000001A04: 81901E10
	s_cmp_lt_i32 s16, 1                                        // 000000001A08: BF048110
	s_cselect_b32 s16, -1, 0                                   // 000000001A0C: 981080C1
	s_add_i32 s17, s17, s27                                    // 000000001A10: 81111B11
	s_add_i32 s28, s28, s27                                    // 000000001A14: 811C1B1C
	s_add_i32 s30, s38, s27                                    // 000000001A18: 811E1B26
	s_add_i32 s31, s41, s27                                    // 000000001A1C: 811F1B29
	s_add_i32 s34, s40, s27                                    // 000000001A20: 81221B28
	s_add_i32 s38, s39, s27                                    // 000000001A24: 81261B27
	s_add_i32 s36, s36, s27                                    // 000000001A28: 81241B24
	s_add_i32 s39, s33, s27                                    // 000000001A2C: 81271B21
	s_add_i32 s40, s29, s27                                    // 000000001A30: 81281B1D
	s_and_b32 s33, s37, s16                                    // 000000001A34: 8B211025
	s_add_i32 s16, s17, 30                                     // 000000001A38: 81109E11
	s_add_i32 s17, s28, 30                                     // 000000001A3C: 81119E1C
	s_add_i32 s27, s30, 30                                     // 000000001A40: 811B9E1E
	s_add_i32 s28, s31, 30                                     // 000000001A44: 811C9E1F
	s_add_i32 s29, s34, 30                                     // 000000001A48: 811D9E22
	s_add_i32 s30, s38, 30                                     // 000000001A4C: 811E9E26
	s_add_i32 s31, s36, 30                                     // 000000001A50: 811F9E24
	s_and_b32 s33, s33, s35                                    // 000000001A54: 8B212321
	s_add_i32 s34, s39, 30                                     // 000000001A58: 81229E27
	s_add_i32 s35, s40, 30                                     // 000000001A5C: 81239E28
	s_and_b32 s0, exec_lo, s0                                  // 000000001A60: 8B00007E
	s_branch 34                                                // 000000001A64: BFA00022 <r_2_4_9_7_4_3_3n1+0x3f0>
	s_clause 0x1                                               // 000000001A68: BF850001
	s_load_b32 s45, s[10:11], null                             // 000000001A6C: F4000B45 F8000000
	s_load_b256 s[48:55], s[10:11], -0x20                      // 000000001A74: F40C0C05 F81FFFE0
	s_add_i32 s1, s1, 4                                        // 000000001A7C: 81018401
	s_add_u32 s10, s10, 0x90                                   // 000000001A80: 800AFF0A 00000090
	s_addc_u32 s11, s11, 0                                     // 000000001A88: 820B800B
	s_cmp_eq_u32 s1, 16                                        // 000000001A8C: BF069001
	s_waitcnt lgkmcnt(0)                                       // 000000001A90: BF89FC07
	v_fmac_f32_e64 v0, s36, s45                                // 000000001A94: D52B0000 00005A24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A9C: BF870091
	v_fmac_f32_e64 v0, s38, s55                                // 000000001AA0: D52B0000 00006E26
	v_fmac_f32_e64 v0, s37, s54                                // 000000001AA8: D52B0000 00006C25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AB0: BF870091
	v_fmac_f32_e64 v0, s40, s53                                // 000000001AB4: D52B0000 00006A28
	v_fmac_f32_e64 v0, s39, s52                                // 000000001ABC: D52B0000 00006827
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AC4: BF870091
	v_fmac_f32_e64 v0, s42, s51                                // 000000001AC8: D52B0000 0000662A
	v_fmac_f32_e64 v0, s41, s50                                // 000000001AD0: D52B0000 00006429
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AD8: BF870091
	v_fmac_f32_e64 v0, s44, s49                                // 000000001ADC: D52B0000 0000622C
	v_fmac_f32_e64 v0, s43, s48                                // 000000001AE4: D52B0000 0000602B
	s_cbranch_scc1 215                                         // 000000001AEC: BFA200D7 <r_2_4_9_7_4_3_3n1+0x74c>
	s_mov_b32 s36, 0                                           // 000000001AF0: BEA40080
	s_mov_b32 vcc_lo, s0                                       // 000000001AF4: BEEA0000
	s_cbranch_vccnz 25                                         // 000000001AF8: BFA40019 <r_2_4_9_7_4_3_3n1+0x460>
	s_mov_b32 s37, 0                                           // 000000001AFC: BEA50080
	s_and_not1_b32 vcc_lo, exec_lo, s12                        // 000000001B00: 916A0C7E
	s_mov_b32 s38, 0                                           // 000000001B04: BEA60080
	s_cbranch_vccz 43                                          // 000000001B08: BFA3002B <r_2_4_9_7_4_3_3n1+0x4b8>
	s_and_not1_b32 vcc_lo, exec_lo, s20                        // 000000001B0C: 916A147E
	s_cbranch_vccz 61                                          // 000000001B10: BFA3003D <r_2_4_9_7_4_3_3n1+0x508>
	s_mov_b32 s39, 0                                           // 000000001B14: BEA70080
	s_and_not1_b32 vcc_lo, exec_lo, s23                        // 000000001B18: 916A177E
	s_mov_b32 s40, 0                                           // 000000001B1C: BEA80080
	s_cbranch_vccz 79                                          // 000000001B20: BFA3004F <r_2_4_9_7_4_3_3n1+0x560>
	s_and_not1_b32 vcc_lo, exec_lo, s24                        // 000000001B24: 916A187E
	s_cbranch_vccz 97                                          // 000000001B28: BFA30061 <r_2_4_9_7_4_3_3n1+0x5b0>
	s_mov_b32 s41, 0                                           // 000000001B2C: BEA90080
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 000000001B30: 916A0D7E
	s_mov_b32 s42, 0                                           // 000000001B34: BEAA0080
	s_cbranch_vccz 115                                         // 000000001B38: BFA30073 <r_2_4_9_7_4_3_3n1+0x608>
	s_and_not1_b32 vcc_lo, exec_lo, s25                        // 000000001B3C: 916A197E
	s_cbranch_vccz 133                                         // 000000001B40: BFA30085 <r_2_4_9_7_4_3_3n1+0x658>
	s_mov_b32 s43, 0                                           // 000000001B44: BEAB0080
	s_and_not1_b32 vcc_lo, exec_lo, s26                        // 000000001B48: 916A1A7E
	s_mov_b32 s44, 0                                           // 000000001B4C: BEAC0080
	s_cbranch_vccz 151                                         // 000000001B50: BFA30097 <r_2_4_9_7_4_3_3n1+0x6b0>
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000001B54: 916A217E
	s_cbranch_vccnz 65475                                      // 000000001B58: BFA4FFC3 <r_2_4_9_7_4_3_3n1+0x368>
	s_branch 168                                               // 000000001B5C: BFA000A8 <r_2_4_9_7_4_3_3n1+0x700>
	s_add_i32 s36, s35, s1                                     // 000000001B60: 81240123
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B64: BF870499
	s_ashr_i32 s37, s36, 31                                    // 000000001B68: 86259F24
	s_lshr_b32 s37, s37, 27                                    // 000000001B6C: 85259B25
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B70: BF870499
	s_add_i32 s37, s36, s37                                    // 000000001B74: 81252524
	s_and_not1_b32 s37, s37, 31                                // 000000001B78: 91259F25
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B7C: BF870499
	s_sub_i32 s36, s36, s37                                    // 000000001B80: 81A42524
	s_mul_i32 s36, s36, 5                                      // 000000001B84: 96248524
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B88: BF870499
	s_ashr_i32 s37, s36, 31                                    // 000000001B8C: 86259F24
	s_lshl_b64 s[36:37], s[36:37], 2                           // 000000001B90: 84A48224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B94: BF870009
	s_add_u32 s36, s3, s36                                     // 000000001B98: 80242403
	s_addc_u32 s37, s9, s37                                    // 000000001B9C: 82252509
	s_load_b32 s36, s[36:37], null                             // 000000001BA0: F4000912 F8000000
	s_mov_b32 s37, 0                                           // 000000001BA8: BEA50080
	s_and_not1_b32 vcc_lo, exec_lo, s12                        // 000000001BAC: 916A0C7E
	s_mov_b32 s38, 0                                           // 000000001BB0: BEA60080
	s_cbranch_vccnz 65493                                      // 000000001BB4: BFA4FFD5 <r_2_4_9_7_4_3_3n1+0x40c>
	s_add_i32 s38, s34, s1                                     // 000000001BB8: 81260122
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BBC: BF870499
	s_ashr_i32 s39, s38, 31                                    // 000000001BC0: 86279F26
	s_lshr_b32 s39, s39, 27                                    // 000000001BC4: 85279B27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BC8: BF870499
	s_add_i32 s39, s38, s39                                    // 000000001BCC: 81272726
	s_and_not1_b32 s39, s39, 31                                // 000000001BD0: 91279F27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BD4: BF870499
	s_sub_i32 s38, s38, s39                                    // 000000001BD8: 81A62726
	s_mul_i32 s38, s38, 5                                      // 000000001BDC: 96268526
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BE0: BF870499
	s_ashr_i32 s39, s38, 31                                    // 000000001BE4: 86279F26
	s_lshl_b64 s[38:39], s[38:39], 2                           // 000000001BE8: 84A68226
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001BEC: BF870009
	s_add_u32 s38, s18, s38                                    // 000000001BF0: 80262612
	s_addc_u32 s39, s19, s39                                   // 000000001BF4: 82272713
	s_load_b32 s38, s[38:39], null                             // 000000001BF8: F4000993 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s20                        // 000000001C00: 916A147E
	s_cbranch_vccnz 65475                                      // 000000001C04: BFA4FFC3 <r_2_4_9_7_4_3_3n1+0x414>
	s_add_i32 s37, s31, s1                                     // 000000001C08: 8125011F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C0C: BF870499
	s_ashr_i32 s39, s37, 31                                    // 000000001C10: 86279F25
	s_lshr_b32 s39, s39, 27                                    // 000000001C14: 85279B27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C18: BF870499
	s_add_i32 s39, s37, s39                                    // 000000001C1C: 81272725
	s_and_not1_b32 s39, s39, 31                                // 000000001C20: 91279F27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C24: BF870499
	s_sub_i32 s37, s37, s39                                    // 000000001C28: 81A52725
	s_mul_i32 s40, s37, 5                                      // 000000001C2C: 96288525
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C30: BF870499
	s_ashr_i32 s41, s40, 31                                    // 000000001C34: 86299F28
	s_lshl_b64 s[40:41], s[40:41], 2                           // 000000001C38: 84A88228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C3C: BF870009
	s_add_u32 s40, s21, s40                                    // 000000001C40: 80282815
	s_addc_u32 s41, s22, s41                                   // 000000001C44: 82292916
	s_load_b32 s37, s[40:41], null                             // 000000001C48: F4000954 F8000000
	s_mov_b32 s39, 0                                           // 000000001C50: BEA70080
	s_and_not1_b32 vcc_lo, exec_lo, s23                        // 000000001C54: 916A177E
	s_mov_b32 s40, 0                                           // 000000001C58: BEA80080
	s_cbranch_vccnz 65457                                      // 000000001C5C: BFA4FFB1 <r_2_4_9_7_4_3_3n1+0x424>
	s_add_i32 s40, s30, s1                                     // 000000001C60: 8128011E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C64: BF870499
	s_ashr_i32 s41, s40, 31                                    // 000000001C68: 86299F28
	s_lshr_b32 s41, s41, 27                                    // 000000001C6C: 85299B29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C70: BF870499
	s_add_i32 s41, s40, s41                                    // 000000001C74: 81292928
	s_and_not1_b32 s41, s41, 31                                // 000000001C78: 91299F29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C7C: BF870499
	s_sub_i32 s40, s40, s41                                    // 000000001C80: 81A82928
	s_mul_i32 s40, s40, 5                                      // 000000001C84: 96288528
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C88: BF870499
	s_ashr_i32 s41, s40, 31                                    // 000000001C8C: 86299F28
	s_lshl_b64 s[40:41], s[40:41], 2                           // 000000001C90: 84A88228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C94: BF870009
	s_add_u32 s40, s3, s40                                     // 000000001C98: 80282803
	s_addc_u32 s41, s9, s41                                    // 000000001C9C: 82292909
	s_load_b32 s40, s[40:41], null                             // 000000001CA0: F4000A14 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s24                        // 000000001CA8: 916A187E
	s_cbranch_vccnz 65439                                      // 000000001CAC: BFA4FF9F <r_2_4_9_7_4_3_3n1+0x42c>
	s_add_i32 s39, s29, s1                                     // 000000001CB0: 8127011D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001CB4: BF870499
	s_ashr_i32 s41, s39, 31                                    // 000000001CB8: 86299F27
	s_lshr_b32 s41, s41, 27                                    // 000000001CBC: 85299B29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001CC0: BF870499
	s_add_i32 s41, s39, s41                                    // 000000001CC4: 81292927
	s_and_not1_b32 s41, s41, 31                                // 000000001CC8: 91299F29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001CCC: BF870499
	s_sub_i32 s39, s39, s41                                    // 000000001CD0: 81A72927
	s_mul_i32 s42, s39, 5                                      // 000000001CD4: 962A8527
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001CD8: BF870499
	s_ashr_i32 s43, s42, 31                                    // 000000001CDC: 862B9F2A
	s_lshl_b64 s[42:43], s[42:43], 2                           // 000000001CE0: 84AA822A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001CE4: BF870009
	s_add_u32 s42, s18, s42                                    // 000000001CE8: 802A2A12
	s_addc_u32 s43, s19, s43                                   // 000000001CEC: 822B2B13
	s_load_b32 s39, s[42:43], null                             // 000000001CF0: F40009D5 F8000000
	s_mov_b32 s41, 0                                           // 000000001CF8: BEA90080
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 000000001CFC: 916A0D7E
	s_mov_b32 s42, 0                                           // 000000001D00: BEAA0080
	s_cbranch_vccnz 65421                                      // 000000001D04: BFA4FF8D <r_2_4_9_7_4_3_3n1+0x43c>
	s_add_i32 s42, s28, s1                                     // 000000001D08: 812A011C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D0C: BF870499
	s_ashr_i32 s43, s42, 31                                    // 000000001D10: 862B9F2A
	s_lshr_b32 s43, s43, 27                                    // 000000001D14: 852B9B2B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D18: BF870499
	s_add_i32 s43, s42, s43                                    // 000000001D1C: 812B2B2A
	s_and_not1_b32 s43, s43, 31                                // 000000001D20: 912B9F2B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D24: BF870499
	s_sub_i32 s42, s42, s43                                    // 000000001D28: 81AA2B2A
	s_mul_i32 s42, s42, 5                                      // 000000001D2C: 962A852A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D30: BF870499
	s_ashr_i32 s43, s42, 31                                    // 000000001D34: 862B9F2A
	s_lshl_b64 s[42:43], s[42:43], 2                           // 000000001D38: 84AA822A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D3C: BF870009
	s_add_u32 s42, s21, s42                                    // 000000001D40: 802A2A15
	s_addc_u32 s43, s22, s43                                   // 000000001D44: 822B2B16
	s_load_b32 s42, s[42:43], null                             // 000000001D48: F4000A95 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s25                        // 000000001D50: 916A197E
	s_cbranch_vccnz 65403                                      // 000000001D54: BFA4FF7B <r_2_4_9_7_4_3_3n1+0x444>
	s_add_i32 s41, s27, s1                                     // 000000001D58: 8129011B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D5C: BF870499
	s_ashr_i32 s43, s41, 31                                    // 000000001D60: 862B9F29
	s_lshr_b32 s43, s43, 27                                    // 000000001D64: 852B9B2B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D68: BF870499
	s_add_i32 s43, s41, s43                                    // 000000001D6C: 812B2B29
	s_and_not1_b32 s43, s43, 31                                // 000000001D70: 912B9F2B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D74: BF870499
	s_sub_i32 s41, s41, s43                                    // 000000001D78: 81A92B29
	s_mul_i32 s44, s41, 5                                      // 000000001D7C: 962C8529
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D80: BF870499
	s_ashr_i32 s45, s44, 31                                    // 000000001D84: 862D9F2C
	s_lshl_b64 s[44:45], s[44:45], 2                           // 000000001D88: 84AC822C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D8C: BF870009
	s_add_u32 s44, s3, s44                                     // 000000001D90: 802C2C03
	s_addc_u32 s45, s9, s45                                    // 000000001D94: 822D2D09
	s_load_b32 s41, s[44:45], null                             // 000000001D98: F4000A56 F8000000
	s_mov_b32 s43, 0                                           // 000000001DA0: BEAB0080
	s_and_not1_b32 vcc_lo, exec_lo, s26                        // 000000001DA4: 916A1A7E
	s_mov_b32 s44, 0                                           // 000000001DA8: BEAC0080
	s_cbranch_vccnz 65385                                      // 000000001DAC: BFA4FF69 <r_2_4_9_7_4_3_3n1+0x454>
	s_add_i32 s44, s17, s1                                     // 000000001DB0: 812C0111
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DB4: BF870499
	s_ashr_i32 s45, s44, 31                                    // 000000001DB8: 862D9F2C
	s_lshr_b32 s45, s45, 27                                    // 000000001DBC: 852D9B2D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DC0: BF870499
	s_add_i32 s45, s44, s45                                    // 000000001DC4: 812D2D2C
	s_and_not1_b32 s45, s45, 31                                // 000000001DC8: 912D9F2D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DCC: BF870499
	s_sub_i32 s44, s44, s45                                    // 000000001DD0: 81AC2D2C
	s_mul_i32 s44, s44, 5                                      // 000000001DD4: 962C852C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DD8: BF870499
	s_ashr_i32 s45, s44, 31                                    // 000000001DDC: 862D9F2C
	s_lshl_b64 s[44:45], s[44:45], 2                           // 000000001DE0: 84AC822C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DE4: BF870009
	s_add_u32 s44, s6, s44                                     // 000000001DE8: 802C2C06
	s_addc_u32 s45, s7, s45                                    // 000000001DEC: 822D2D07
	s_load_b32 s44, s[44:45], null                             // 000000001DF0: F4000B16 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000001DF8: 916A217E
	s_cbranch_vccnz 65306                                      // 000000001DFC: BFA4FF1A <r_2_4_9_7_4_3_3n1+0x368>
	s_add_i32 s43, s16, s1                                     // 000000001E00: 812B0110
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E04: BF870499
	s_ashr_i32 s45, s43, 31                                    // 000000001E08: 862D9F2B
	s_lshr_b32 s45, s45, 27                                    // 000000001E0C: 852D9B2D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E10: BF870499
	s_add_i32 s45, s43, s45                                    // 000000001E14: 812D2D2B
	s_and_not1_b32 s45, s45, 31                                // 000000001E18: 912D9F2D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E1C: BF870499
	s_sub_i32 s43, s43, s45                                    // 000000001E20: 81AB2D2B
	s_mul_i32 s46, s43, 5                                      // 000000001E24: 962E852B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E28: BF870499
	s_ashr_i32 s47, s46, 31                                    // 000000001E2C: 862F9F2E
	s_lshl_b64 s[46:47], s[46:47], 2                           // 000000001E30: 84AE822E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E34: BF870009
	s_add_u32 s46, s21, s46                                    // 000000001E38: 802E2E15
	s_addc_u32 s47, s22, s47                                   // 000000001E3C: 822F2F16
	s_load_b32 s43, s[46:47], null                             // 000000001E40: F4000AD7 F8000000
	s_branch 65287                                             // 000000001E48: BFA0FF07 <r_2_4_9_7_4_3_3n1+0x368>
	s_mul_i32 s0, s15, 0xfc                                    // 000000001E4C: 9600FF0F 000000FC
	s_mul_i32 s6, s14, 63                                      // 000000001E54: 9606BF0E
	s_ashr_i32 s1, s0, 31                                      // 000000001E58: 86019F00
	v_dual_max_f32 v0, v0, v0 :: v_dual_mov_b32 v1, 0          // 000000001E5C: CA900100 00000080
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001E64: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001E68: BF8704D9
	s_add_u32 s3, s4, s0                                       // 000000001E6C: 80030004
	s_addc_u32 s4, s5, s1                                      // 000000001E70: 82040105
	s_ashr_i32 s7, s6, 31                                      // 000000001E74: 86079F06
	v_max_f32_e32 v0, 0, v0                                    // 000000001E78: 20000080
	s_lshl_b64 s[0:1], s[6:7], 2                               // 000000001E7C: 84808206
	s_add_u32 s3, s3, s0                                       // 000000001E80: 80030003
	s_addc_u32 s4, s4, s1                                      // 000000001E84: 82040104
	s_ashr_i32 s9, s8, 31                                      // 000000001E88: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E8C: BF870499
	s_lshl_b64 s[0:1], s[8:9], 2                               // 000000001E90: 84808208
	s_add_u32 s5, s3, s0                                       // 000000001E94: 80050003
	s_addc_u32 s4, s4, s1                                      // 000000001E98: 82040104
	s_ashr_i32 s3, s2, 31                                      // 000000001E9C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001EA0: BF870499
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001EA4: 84808202
	s_add_u32 s0, s5, s0                                       // 000000001EA8: 80000005
	s_addc_u32 s1, s4, s1                                      // 000000001EAC: 82010104
	global_store_b32 v1, v0, s[0:1]                            // 000000001EB0: DC6A0000 00000001
	s_nop 0                                                    // 000000001EB8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001EBC: BFB60003
	s_endpgm                                                   // 000000001EC0: BFB00000
