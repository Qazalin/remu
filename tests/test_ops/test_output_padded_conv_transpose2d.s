
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001800 <r_2_4_14_16_4_3_3>:
	s_load_b256 s[4:11], s[0:1], null                          // 000000001800: F40C0100 F8000000
	s_ashr_i32 s0, s13, 31                                     // 000000001808: 86009F0D
	s_mov_b32 s2, s15                                          // 00000000180C: BE82000F
	s_lshr_b32 s1, s0, 28                                      // 000000001810: 85019C00
	s_ashr_i32 s15, s14, 31                                    // 000000001814: 860F9F0E
	s_add_i32 s1, s13, s1                                      // 000000001818: 8101010D
	s_lshl_b64 s[16:17], s[14:15], 2                           // 00000000181C: 8490820E
	s_and_b32 s12, s1, -16                                     // 000000001820: 8B0CD001
	s_mul_i32 s0, s14, 9                                       // 000000001824: 9600890E
	s_ashr_i32 s3, s1, 4                                       // 000000001828: 86038401
	s_sub_i32 s12, s13, s12                                    // 00000000182C: 818C0C0D
	s_waitcnt lgkmcnt(0)                                       // 000000001830: BF89FC07
	s_add_u32 s10, s10, s16                                    // 000000001834: 800A100A
	s_addc_u32 s11, s11, s17                                   // 000000001838: 820B110B
	s_ashr_i32 s1, s0, 31                                      // 00000000183C: 86019F00
	s_load_b32 s10, s[10:11], null                             // 000000001840: F4000285 F8000000
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001848: 84808200
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000184C: BF870009
	s_add_u32 s0, s8, s0                                       // 000000001850: 80000008
	s_addc_u32 s1, s9, s1                                      // 000000001854: 82010109
	s_add_u32 s8, s0, 32                                       // 000000001858: 8008A000
	s_addc_u32 s9, s1, 0                                       // 00000000185C: 82098001
	s_add_i32 s0, s12, 13                                      // 000000001860: 81008D0C
	s_add_i32 s24, s3, 1                                       // 000000001864: 81188103
	s_mul_hi_i32 s1, s0, 0x88888889                            // 000000001868: 9701FF00 88888889
	s_add_i32 s15, s12, 1                                      // 000000001870: 810F810C
	s_add_i32 s1, s1, s0                                       // 000000001874: 81010001
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001878: BF8704A9
	s_lshr_b32 s0, s1, 31                                      // 00000000187C: 85009F01
	s_ashr_i32 s1, s1, 3                                       // 000000001880: 86018301
	s_add_i32 s29, s1, s0                                      // 000000001884: 811D0001
	s_add_i32 s0, s12, -2                                      // 000000001888: 8100C20C
	s_add_i32 s1, s24, s29                                     // 00000000188C: 81011D18
	s_cmp_lt_u32 s0, 13                                        // 000000001890: BF0A8D00
	s_cselect_b32 s30, -1, 0                                   // 000000001894: 981E80C1
	s_sub_i32 s0, s13, 32                                      // 000000001898: 8180A00D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 00000000189C: BF8704C9
	s_cmpk_lt_u32 s0, 0xb0                                     // 0000000018A0: B68000B0
	s_mul_hi_i32 s0, s15, 0x55555556                           // 0000000018A4: 9700FF0F 55555556
	s_cselect_b32 s25, -1, 0                                   // 0000000018AC: 981980C1
	s_lshr_b32 s16, s0, 31                                     // 0000000018B0: 85109F00
	s_add_i32 s0, s0, s16                                      // 0000000018B4: 81001000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018B8: BF870499
	s_mul_i32 s16, s0, 3                                       // 0000000018BC: 96108300
	s_sub_i32 s15, s15, s16                                    // 0000000018C0: 818F100F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000018C4: BF8704B9
	s_cmp_lt_i32 s15, 1                                        // 0000000018C8: BF04810F
	s_cselect_b32 s31, -1, 0                                   // 0000000018CC: 981F80C1
	s_lshr_b32 s15, s1, 31                                     // 0000000018D0: 850F9F01
	s_add_i32 s15, s1, s15                                     // 0000000018D4: 810F0F01
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000018D8: BF8704B9
	s_and_b32 s16, s15, -2                                     // 0000000018DC: 8B10C20F
	s_ashr_i32 s33, s15, 1                                     // 0000000018E0: 8621810F
	s_sub_i32 s1, s1, s16                                      // 0000000018E4: 81811001
	s_cmp_lt_i32 s1, 1                                         // 0000000018E8: BF048101
	s_cselect_b32 s15, -1, 0                                   // 0000000018EC: 980F80C1
	s_add_i32 s0, s0, 4                                        // 0000000018F0: 81008400
	s_and_b32 s15, s25, s15                                    // 0000000018F4: 8B0F0F19
	s_mul_hi_u32 s1, s0, 0xcccccccd                            // 0000000018F8: 9681FF00 CCCCCCCD
	s_and_b32 s15, s15, s30                                    // 000000001900: 8B0F1E0F
	s_lshr_b32 s16, s1, 2                                      // 000000001904: 85108201
	s_mov_b32 s1, 0                                            // 000000001908: BE810080
	s_mul_i32 s16, s16, 5                                      // 00000000190C: 96108510
	s_and_b32 s39, s15, s31                                    // 000000001910: 8B271F0F
	s_sub_i32 s0, s0, s16                                      // 000000001914: 81801000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001918: BF870499
	s_lshl_b64 s[18:19], s[0:1], 2                             // 00000000191C: 84928200
	s_add_u32 s15, s6, s18                                     // 000000001920: 800F1206
	s_addc_u32 s20, s7, s19                                    // 000000001924: 82141307
	s_add_i32 s0, s12, 14                                      // 000000001928: 81008E0C
	s_add_i32 s17, s12, 2                                      // 00000000192C: 8111820C
	s_mul_hi_i32 s16, s0, 0x88888889                           // 000000001930: 9710FF00 88888889
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001938: BF870499
	s_add_i32 s16, s16, s0                                     // 00000000193C: 81100010
	s_lshr_b32 s0, s16, 31                                     // 000000001940: 85009F10
	s_ashr_i32 s16, s16, 3                                     // 000000001944: 86108310
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001948: BF870009
	s_add_i32 s34, s16, s0                                     // 00000000194C: 81220010
	s_add_i32 s0, s12, -1                                      // 000000001950: 8100C10C
	s_add_i32 s16, s24, s34                                    // 000000001954: 81102218
	s_cmp_lt_u32 s0, 13                                        // 000000001958: BF0A8D00
	s_mul_hi_i32 s0, s17, 0x55555556                           // 00000000195C: 9700FF11 55555556
	s_cselect_b32 s35, -1, 0                                   // 000000001964: 982380C1
	s_lshr_b32 s21, s0, 31                                     // 000000001968: 85159F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000196C: BF870499
	s_add_i32 s0, s0, s21                                      // 000000001970: 81001500
	s_mul_i32 s21, s0, 3                                       // 000000001974: 96158300
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001978: BF870499
	s_sub_i32 s17, s17, s21                                    // 00000000197C: 81911511
	s_cmp_lt_i32 s17, 1                                        // 000000001980: BF048111
	s_cselect_b32 s36, -1, 0                                   // 000000001984: 982480C1
	s_lshr_b32 s17, s16, 31                                    // 000000001988: 85119F10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000198C: BF870499
	s_add_i32 s17, s16, s17                                    // 000000001990: 81111110
	s_and_b32 s21, s17, -2                                     // 000000001994: 8B15C211
	s_ashr_i32 s37, s17, 1                                     // 000000001998: 86258111
	s_sub_i32 s16, s16, s21                                    // 00000000199C: 81901510
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019A0: BF870009
	s_cmp_lt_i32 s16, 1                                        // 0000000019A4: BF048110
	s_cselect_b32 s16, -1, 0                                   // 0000000019A8: 981080C1
	s_add_i32 s0, s0, 4                                        // 0000000019AC: 81008400
	s_and_b32 s16, s25, s16                                    // 0000000019B0: 8B101019
	s_mul_hi_u32 s17, s0, 0xcccccccd                           // 0000000019B4: 9691FF00 CCCCCCCD
	s_and_b32 s21, s16, s35                                    // 0000000019BC: 8B152310
	s_lshr_b32 s17, s17, 2                                     // 0000000019C0: 85118211
	s_and_b32 s21, s21, s36                                    // 0000000019C4: 8B152415
	s_mul_i32 s17, s17, 5                                      // 0000000019C8: 96118511
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019CC: BF870499
	s_sub_i32 s0, s0, s17                                      // 0000000019D0: 81801100
	s_lshl_b64 s[16:17], s[0:1], 2                             // 0000000019D4: 84908200
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019D8: BF870009
	s_add_u32 s22, s6, s16                                     // 0000000019DC: 80161006
	s_addc_u32 s23, s7, s17                                    // 0000000019E0: 82171107
	s_add_i32 s0, s12, 15                                      // 0000000019E4: 81008F0C
	s_add_i32 s26, s12, 3                                      // 0000000019E8: 811A830C
	s_mul_hi_u32 s0, s0, 0x88888889                            // 0000000019EC: 9680FF00 88888889
	v_and_b32_e64 v0, 0xff, s26                                // 0000000019F4: D51B0000 000034FF 000000FF
	s_lshr_b32 s38, s0, 3                                      // 000000001A00: 85268300
	s_mul_hi_i32 s0, s26, 0x55555556                           // 000000001A04: 9700FF1A 55555556
	s_add_i32 s24, s24, s38                                    // 000000001A0C: 81182618
	s_cmp_lt_u32 s12, 13                                       // 000000001A10: BF0A8D0C
	v_mul_lo_u16 v1, 0xab, v0                                  // 000000001A14: D7050001 000200FF 000000AB
	s_cselect_b32 s40, -1, 0                                   // 000000001A20: 982880C1
	s_lshr_b32 s27, s0, 31                                     // 000000001A24: 851B9F00
	v_cmp_gt_u16_e32 vcc_lo, 3, v0                             // 000000001A28: 7C780083
	s_add_i32 s0, s0, s27                                      // 000000001A2C: 81001B00
	v_lshrrev_b16 v1, 9, v1                                    // 000000001A30: D7390001 00020289
	s_mul_i32 s0, s0, 3                                        // 000000001A38: 96008300
	v_mov_b32_e32 v0, 0                                        // 000000001A3C: 7E000280
	s_sub_i32 s0, s26, s0                                      // 000000001A40: 8180001A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001A44: BF8704C9
	s_cmp_lt_i32 s0, 1                                         // 000000001A48: BF048100
	v_readfirstlane_b32 s27, v1                                // 000000001A4C: 7E360501
	s_cselect_b32 s41, -1, 0                                   // 000000001A50: 982980C1
	s_lshr_b32 s0, s24, 31                                     // 000000001A54: 85009F18
	s_add_i32 s0, s24, s0                                      // 000000001A58: 81000018
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001A5C: BF8704B9
	s_and_b32 s26, s0, -2                                      // 000000001A60: 8B1AC200
	s_ashr_i32 s42, s0, 1                                      // 000000001A64: 862A8100
	s_sub_i32 s24, s24, s26                                    // 000000001A68: 81981A18
	s_cmp_lt_i32 s24, 1                                        // 000000001A6C: BF048118
	s_cselect_b32 s0, -1, 0                                    // 000000001A70: 980080C1
	s_add_i32 s27, s27, 4                                      // 000000001A74: 811B841B
	s_and_b32 s0, s25, s0                                      // 000000001A78: 8B000019
	s_and_b32 s25, s27, 0xff                                   // 000000001A7C: 8B19FF1B 000000FF
	s_and_b32 s0, s0, s40                                      // 000000001A84: 8B002800
	s_add_i32 s26, s25, -5                                     // 000000001A88: 811AC519
	s_and_b32 s24, s0, s41                                     // 000000001A8C: 8B182900
	s_and_b32 s0, vcc_lo, exec_lo                              // 000000001A90: 8B007E6A
	s_cselect_b32 s0, s25, s26                                 // 000000001A94: 98001A19
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001A98: BF870499
	s_lshl_b64 s[26:27], s[0:1], 2                             // 000000001A9C: 849A8200
	s_add_u32 s25, s6, s26                                     // 000000001AA0: 80191A06
	s_addc_u32 s26, s7, s27                                    // 000000001AA4: 821A1B07
	s_add_i32 s0, s3, 2                                        // 000000001AA8: 81008203
	s_add_i32 s27, s13, -16                                    // 000000001AAC: 811BD00D
	s_add_i32 s28, s0, s29                                     // 000000001AB0: 811C1D00
	s_cmpk_lt_u32 s27, 0xb0                                    // 000000001AB4: B69B00B0
	s_cselect_b32 s43, -1, 0                                   // 000000001AB8: 982B80C1
	s_lshr_b32 s27, s28, 31                                    // 000000001ABC: 851B9F1C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001AC0: BF870499
	s_add_i32 s27, s28, s27                                    // 000000001AC4: 811B1B1C
	s_and_b32 s44, s27, -2                                     // 000000001AC8: 8B2CC21B
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001ACC: BF870009
	s_sub_i32 s28, s28, s44                                    // 000000001AD0: 819C2C1C
	s_ashr_i32 s44, s27, 1                                     // 000000001AD4: 862C811B
	s_cmp_lt_i32 s28, 1                                        // 000000001AD8: BF04811C
	s_cselect_b32 s27, -1, 0                                   // 000000001ADC: 981B80C1
	s_add_i32 s28, s0, s34                                     // 000000001AE0: 811C2200
	s_and_b32 s27, s43, s27                                    // 000000001AE4: 8B1B1B2B
	s_lshr_b32 s45, s28, 31                                    // 000000001AE8: 852D9F1C
	s_and_b32 s27, s27, s30                                    // 000000001AEC: 8B1B1E1B
	s_add_i32 s45, s28, s45                                    // 000000001AF0: 812D2D1C
	s_and_b32 s27, s27, s31                                    // 000000001AF4: 8B1B1F1B
	s_and_b32 s46, s45, -2                                     // 000000001AF8: 8B2EC22D
	s_ashr_i32 s45, s45, 1                                     // 000000001AFC: 862D812D
	s_sub_i32 s28, s28, s46                                    // 000000001B00: 819C2E1C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B04: BF870009
	s_cmp_lt_i32 s28, 1                                        // 000000001B08: BF04811C
	s_cselect_b32 s28, -1, 0                                   // 000000001B0C: 981C80C1
	s_add_i32 s0, s0, s38                                      // 000000001B10: 81002600
	s_and_b32 s28, s43, s28                                    // 000000001B14: 8B1C1C2B
	s_lshr_b32 s46, s0, 31                                     // 000000001B18: 852E9F00
	s_and_b32 s28, s28, s35                                    // 000000001B1C: 8B1C231C
	s_add_i32 s46, s0, s46                                     // 000000001B20: 812E2E00
	s_and_b32 s28, s28, s36                                    // 000000001B24: 8B1C241C
	s_and_b32 s47, s46, -2                                     // 000000001B28: 8B2FC22E
	s_ashr_i32 s46, s46, 1                                     // 000000001B2C: 862E812E
	s_sub_i32 s0, s0, s47                                      // 000000001B30: 81802F00
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B34: BF870009
	s_cmp_lt_i32 s0, 1                                         // 000000001B38: BF048100
	s_cselect_b32 s0, -1, 0                                    // 000000001B3C: 980080C1
	s_add_i32 s47, s3, 3                                       // 000000001B40: 812F8303
	s_and_b32 s0, s43, s0                                      // 000000001B44: 8B00002B
	s_add_i32 s43, s13, 15                                     // 000000001B48: 812B8F0D
	s_and_b32 s0, s0, s40                                      // 000000001B4C: 8B002800
	s_add_i32 s29, s47, s29                                    // 000000001B50: 811D1D2F
	s_and_b32 s13, s0, s41                                     // 000000001B54: 8B0D2900
	s_cmpk_lt_u32 s43, 0xbf                                    // 000000001B58: B6AB00BF
	s_cselect_b32 s0, -1, 0                                    // 000000001B5C: 980080C1
	s_lshr_b32 s43, s29, 31                                    // 000000001B60: 852B9F1D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B64: BF870499
	s_add_i32 s43, s29, s43                                    // 000000001B68: 812B2B1D
	s_and_b32 s48, s43, -2                                     // 000000001B6C: 8B30C22B
	s_ashr_i32 s43, s43, 1                                     // 000000001B70: 862B812B
	s_sub_i32 s29, s29, s48                                    // 000000001B74: 819D301D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001B78: BF8704A9
	s_cmp_lt_i32 s29, 1                                        // 000000001B7C: BF04811D
	s_cselect_b32 s29, -1, 0                                   // 000000001B80: 981D80C1
	s_and_b32 s29, s0, s29                                     // 000000001B84: 8B1D1D00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B88: BF870499
	s_and_b32 s29, s29, s30                                    // 000000001B8C: 8B1D1E1D
	s_and_b32 s29, s29, s31                                    // 000000001B90: 8B1D1F1D
	s_add_u32 s18, s6, s18                                     // 000000001B94: 80121206
	s_addc_u32 s19, s7, s19                                    // 000000001B98: 82131307
	s_add_i32 s30, s47, s34                                    // 000000001B9C: 811E222F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BA0: BF870499
	s_lshr_b32 s31, s30, 31                                    // 000000001BA4: 851F9F1E
	s_add_i32 s31, s30, s31                                    // 000000001BA8: 811F1F1E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001BAC: BF8704B9
	s_and_b32 s34, s31, -2                                     // 000000001BB0: 8B22C21F
	s_ashr_i32 s31, s31, 1                                     // 000000001BB4: 861F811F
	s_sub_i32 s30, s30, s34                                    // 000000001BB8: 819E221E
	s_cmp_lt_i32 s30, 1                                        // 000000001BBC: BF04811E
	s_mul_i32 s30, s2, 24                                      // 000000001BC0: 961E9802
	s_cselect_b32 s34, -1, 0                                   // 000000001BC4: 982280C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BC8: BF870499
	s_and_b32 s11, s0, s34                                     // 000000001BCC: 8B0B2200
	s_and_b32 s11, s11, s35                                    // 000000001BD0: 8B0B230B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001BD4: BF8704C9
	s_and_b32 s11, s11, s36                                    // 000000001BD8: 8B0B240B
	s_add_u32 s6, s6, s16                                      // 000000001BDC: 80061006
	s_addc_u32 s7, s7, s17                                     // 000000001BE0: 82071107
	s_add_i32 s16, s47, s38                                    // 000000001BE4: 8110262F
	s_lshr_b32 s17, s16, 31                                    // 000000001BE8: 85119F10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BEC: BF870499
	s_add_i32 s17, s16, s17                                    // 000000001BF0: 81111110
	s_and_b32 s34, s17, -2                                     // 000000001BF4: 8B22C211
	s_ashr_i32 s17, s17, 1                                     // 000000001BF8: 86118111
	s_sub_i32 s16, s16, s34                                    // 000000001BFC: 81902210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C00: BF870009
	s_cmp_lt_i32 s16, 1                                        // 000000001C04: BF048110
	s_cselect_b32 s16, -1, 0                                   // 000000001C08: 981080C1
	s_add_i32 s33, s33, s30                                    // 000000001C0C: 81211E21
	s_and_b32 s0, s0, s16                                      // 000000001C10: 8B001000
	s_add_i32 s17, s17, s30                                    // 000000001C14: 81111E11
	s_add_i32 s31, s31, s30                                    // 000000001C18: 811F1E1F
	s_add_i32 s34, s43, s30                                    // 000000001C1C: 81221E2B
	s_add_i32 s35, s46, s30                                    // 000000001C20: 81231E2E
	s_add_i32 s36, s45, s30                                    // 000000001C24: 81241E2D
	s_add_i32 s38, s44, s30                                    // 000000001C28: 81261E2C
	s_add_i32 s42, s42, s30                                    // 000000001C2C: 812A1E2A
	s_add_i32 s43, s37, s30                                    // 000000001C30: 812B1E25
	s_and_b32 s0, s0, s40                                      // 000000001C34: 8B002800
	s_add_i32 s16, s33, 46                                     // 000000001C38: 8110AE21
	s_add_i32 s17, s17, 46                                     // 000000001C3C: 8111AE11
	s_add_i32 s30, s31, 46                                     // 000000001C40: 811EAE1F
	s_add_i32 s31, s34, 46                                     // 000000001C44: 811FAE22
	s_add_i32 s33, s35, 46                                     // 000000001C48: 8121AE23
	s_add_i32 s34, s36, 46                                     // 000000001C4C: 8122AE24
	s_add_i32 s35, s38, 46                                     // 000000001C50: 8123AE26
	s_and_b32 s36, s0, s41                                     // 000000001C54: 8B242900
	s_add_i32 s37, s42, 46                                     // 000000001C58: 8125AE2A
	s_add_i32 s38, s43, 46                                     // 000000001C5C: 8126AE2B
	s_and_b32 s0, exec_lo, s39                                 // 000000001C60: 8B00277E
	s_branch 34                                                // 000000001C64: BFA00022 <r_2_4_14_16_4_3_3+0x4f0>
	s_clause 0x1                                               // 000000001C68: BF850001
	s_load_b32 s56, s[8:9], null                               // 000000001C6C: F4000E04 F8000000
	s_load_b256 s[48:55], s[8:9], -0x20                        // 000000001C74: F40C0C04 F81FFFE0
	s_add_i32 s1, s1, 6                                        // 000000001C7C: 81018601
	s_add_u32 s8, s8, 0x90                                     // 000000001C80: 8008FF08 00000090
	s_addc_u32 s9, s9, 0                                       // 000000001C88: 82098009
	s_cmp_eq_u32 s1, 24                                        // 000000001C8C: BF069801
	s_waitcnt lgkmcnt(0)                                       // 000000001C90: BF89FC07
	v_fmac_f32_e64 v0, s39, s56                                // 000000001C94: D52B0000 00007027
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C9C: BF870091
	v_fmac_f32_e64 v0, s41, s55                                // 000000001CA0: D52B0000 00006E29
	v_fmac_f32_e64 v0, s40, s54                                // 000000001CA8: D52B0000 00006C28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CB0: BF870091
	v_fmac_f32_e64 v0, s43, s53                                // 000000001CB4: D52B0000 00006A2B
	v_fmac_f32_e64 v0, s42, s52                                // 000000001CBC: D52B0000 0000682A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CC4: BF870091
	v_fmac_f32_e64 v0, s45, s51                                // 000000001CC8: D52B0000 0000662D
	v_fmac_f32_e64 v0, s44, s50                                // 000000001CD0: D52B0000 0000642C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CD8: BF870091
	v_fmac_f32_e64 v0, s47, s49                                // 000000001CDC: D52B0000 0000622F
	v_fmac_f32_e64 v0, s46, s48                                // 000000001CE4: D52B0000 0000602E
	s_cbranch_scc1 233                                         // 000000001CEC: BFA200E9 <r_2_4_14_16_4_3_3+0x894>
	s_mov_b32 s39, 0                                           // 000000001CF0: BEA70080
	s_mov_b32 vcc_lo, s0                                       // 000000001CF4: BEEA0000
	s_cbranch_vccnz 25                                         // 000000001CF8: BFA40019 <r_2_4_14_16_4_3_3+0x560>
	s_mov_b32 s40, 0                                           // 000000001CFC: BEA80080
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 000000001D00: 916A157E
	s_mov_b32 s41, 0                                           // 000000001D04: BEA90080
	s_cbranch_vccz 45                                          // 000000001D08: BFA3002D <r_2_4_14_16_4_3_3+0x5c0>
	s_and_not1_b32 vcc_lo, exec_lo, s24                        // 000000001D0C: 916A187E
	s_cbranch_vccz 65                                          // 000000001D10: BFA30041 <r_2_4_14_16_4_3_3+0x618>
	s_mov_b32 s42, 0                                           // 000000001D14: BEAA0080
	s_and_not1_b32 vcc_lo, exec_lo, s27                        // 000000001D18: 916A1B7E
	s_mov_b32 s43, 0                                           // 000000001D1C: BEAB0080
	s_cbranch_vccz 85                                          // 000000001D20: BFA30055 <r_2_4_14_16_4_3_3+0x678>
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 000000001D24: 916A1C7E
	s_cbranch_vccz 105                                         // 000000001D28: BFA30069 <r_2_4_14_16_4_3_3+0x6d0>
	s_mov_b32 s44, 0                                           // 000000001D2C: BEAC0080
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 000000001D30: 916A0D7E
	s_mov_b32 s45, 0                                           // 000000001D34: BEAD0080
	s_cbranch_vccz 125                                         // 000000001D38: BFA3007D <r_2_4_14_16_4_3_3+0x730>
	s_and_not1_b32 vcc_lo, exec_lo, s29                        // 000000001D3C: 916A1D7E
	s_cbranch_vccz 145                                         // 000000001D40: BFA30091 <r_2_4_14_16_4_3_3+0x788>
	s_mov_b32 s46, 0                                           // 000000001D44: BEAE0080
	s_and_not1_b32 vcc_lo, exec_lo, s11                        // 000000001D48: 916A0B7E
	s_mov_b32 s47, 0                                           // 000000001D4C: BEAF0080
	s_cbranch_vccz 165                                         // 000000001D50: BFA300A5 <r_2_4_14_16_4_3_3+0x7e8>
	s_and_not1_b32 vcc_lo, exec_lo, s36                        // 000000001D54: 916A247E
	s_cbranch_vccnz 65475                                      // 000000001D58: BFA4FFC3 <r_2_4_14_16_4_3_3+0x468>
	s_branch 184                                               // 000000001D5C: BFA000B8 <r_2_4_14_16_4_3_3+0x840>
	s_add_i32 s39, s16, s1                                     // 000000001D60: 81270110
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D64: BF870499
	s_mul_hi_i32 s40, s39, 0x2aaaaaab                          // 000000001D68: 9728FF27 2AAAAAAB
	s_lshr_b32 s41, s40, 31                                    // 000000001D70: 85299F28
	s_ashr_i32 s40, s40, 3                                     // 000000001D74: 86288328
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D78: BF870499
	s_add_i32 s40, s40, s41                                    // 000000001D7C: 81282928
	s_mul_i32 s40, s40, 48                                     // 000000001D80: 9628B028
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D84: BF870499
	s_sub_i32 s39, s39, s40                                    // 000000001D88: 81A72827
	s_mul_i32 s40, s39, 5                                      // 000000001D8C: 96288527
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D90: BF870499
	s_ashr_i32 s41, s40, 31                                    // 000000001D94: 86299F28
	s_lshl_b64 s[40:41], s[40:41], 2                           // 000000001D98: 84A88228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D9C: BF870009
	s_add_u32 s40, s15, s40                                    // 000000001DA0: 8028280F
	s_addc_u32 s41, s20, s41                                   // 000000001DA4: 82292914
	s_load_b32 s39, s[40:41], null                             // 000000001DA8: F40009D4 F8000000
	s_mov_b32 s40, 0                                           // 000000001DB0: BEA80080
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 000000001DB4: 916A157E
	s_mov_b32 s41, 0                                           // 000000001DB8: BEA90080
	s_cbranch_vccnz 65491                                      // 000000001DBC: BFA4FFD3 <r_2_4_14_16_4_3_3+0x50c>
	s_add_i32 s41, s38, s1                                     // 000000001DC0: 81290126
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DC4: BF870499
	s_mul_hi_i32 s42, s41, 0x2aaaaaab                          // 000000001DC8: 972AFF29 2AAAAAAB
	s_lshr_b32 s43, s42, 31                                    // 000000001DD0: 852B9F2A
	s_ashr_i32 s42, s42, 3                                     // 000000001DD4: 862A832A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DD8: BF870499
	s_add_i32 s42, s42, s43                                    // 000000001DDC: 812A2B2A
	s_mul_i32 s42, s42, 48                                     // 000000001DE0: 962AB02A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DE4: BF870499
	s_sub_i32 s41, s41, s42                                    // 000000001DE8: 81A92A29
	s_mul_i32 s42, s41, 5                                      // 000000001DEC: 962A8529
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DF0: BF870499
	s_ashr_i32 s43, s42, 31                                    // 000000001DF4: 862B9F2A
	s_lshl_b64 s[42:43], s[42:43], 2                           // 000000001DF8: 84AA822A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DFC: BF870009
	s_add_u32 s42, s22, s42                                    // 000000001E00: 802A2A16
	s_addc_u32 s43, s23, s43                                   // 000000001E04: 822B2B17
	s_load_b32 s41, s[42:43], null                             // 000000001E08: F4000A55 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s24                        // 000000001E10: 916A187E
	s_cbranch_vccnz 65471                                      // 000000001E14: BFA4FFBF <r_2_4_14_16_4_3_3+0x514>
	s_add_i32 s40, s37, s1                                     // 000000001E18: 81280125
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E1C: BF870499
	s_mul_hi_i32 s42, s40, 0x2aaaaaab                          // 000000001E20: 972AFF28 2AAAAAAB
	s_lshr_b32 s43, s42, 31                                    // 000000001E28: 852B9F2A
	s_ashr_i32 s42, s42, 3                                     // 000000001E2C: 862A832A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E30: BF870499
	s_add_i32 s42, s42, s43                                    // 000000001E34: 812A2B2A
	s_mul_i32 s42, s42, 48                                     // 000000001E38: 962AB02A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E3C: BF870499
	s_sub_i32 s40, s40, s42                                    // 000000001E40: 81A82A28
	s_mul_i32 s42, s40, 5                                      // 000000001E44: 962A8528
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E48: BF870499
	s_ashr_i32 s43, s42, 31                                    // 000000001E4C: 862B9F2A
	s_lshl_b64 s[42:43], s[42:43], 2                           // 000000001E50: 84AA822A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E54: BF870009
	s_add_u32 s42, s25, s42                                    // 000000001E58: 802A2A19
	s_addc_u32 s43, s26, s43                                   // 000000001E5C: 822B2B1A
	s_load_b32 s40, s[42:43], null                             // 000000001E60: F4000A15 F8000000
	s_mov_b32 s42, 0                                           // 000000001E68: BEAA0080
	s_and_not1_b32 vcc_lo, exec_lo, s27                        // 000000001E6C: 916A1B7E
	s_mov_b32 s43, 0                                           // 000000001E70: BEAB0080
	s_cbranch_vccnz 65451                                      // 000000001E74: BFA4FFAB <r_2_4_14_16_4_3_3+0x524>
	s_add_i32 s43, s35, s1                                     // 000000001E78: 812B0123
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E7C: BF870499
	s_mul_hi_i32 s44, s43, 0x2aaaaaab                          // 000000001E80: 972CFF2B 2AAAAAAB
	s_lshr_b32 s45, s44, 31                                    // 000000001E88: 852D9F2C
	s_ashr_i32 s44, s44, 3                                     // 000000001E8C: 862C832C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E90: BF870499
	s_add_i32 s44, s44, s45                                    // 000000001E94: 812C2D2C
	s_mul_i32 s44, s44, 48                                     // 000000001E98: 962CB02C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E9C: BF870499
	s_sub_i32 s43, s43, s44                                    // 000000001EA0: 81AB2C2B
	s_mul_i32 s44, s43, 5                                      // 000000001EA4: 962C852B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001EA8: BF870499
	s_ashr_i32 s45, s44, 31                                    // 000000001EAC: 862D9F2C
	s_lshl_b64 s[44:45], s[44:45], 2                           // 000000001EB0: 84AC822C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001EB4: BF870009
	s_add_u32 s44, s15, s44                                    // 000000001EB8: 802C2C0F
	s_addc_u32 s45, s20, s45                                   // 000000001EBC: 822D2D14
	s_load_b32 s43, s[44:45], null                             // 000000001EC0: F4000AD6 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 000000001EC8: 916A1C7E
	s_cbranch_vccnz 65431                                      // 000000001ECC: BFA4FF97 <r_2_4_14_16_4_3_3+0x52c>
	s_add_i32 s42, s34, s1                                     // 000000001ED0: 812A0122
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001ED4: BF870499
	s_mul_hi_i32 s44, s42, 0x2aaaaaab                          // 000000001ED8: 972CFF2A 2AAAAAAB
	s_lshr_b32 s45, s44, 31                                    // 000000001EE0: 852D9F2C
	s_ashr_i32 s44, s44, 3                                     // 000000001EE4: 862C832C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001EE8: BF870499
	s_add_i32 s44, s44, s45                                    // 000000001EEC: 812C2D2C
	s_mul_i32 s44, s44, 48                                     // 000000001EF0: 962CB02C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001EF4: BF870499
	s_sub_i32 s42, s42, s44                                    // 000000001EF8: 81AA2C2A
	s_mul_i32 s44, s42, 5                                      // 000000001EFC: 962C852A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F00: BF870499
	s_ashr_i32 s45, s44, 31                                    // 000000001F04: 862D9F2C
	s_lshl_b64 s[44:45], s[44:45], 2                           // 000000001F08: 84AC822C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F0C: BF870009
	s_add_u32 s44, s22, s44                                    // 000000001F10: 802C2C16
	s_addc_u32 s45, s23, s45                                   // 000000001F14: 822D2D17
	s_load_b32 s42, s[44:45], null                             // 000000001F18: F4000A96 F8000000
	s_mov_b32 s44, 0                                           // 000000001F20: BEAC0080
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 000000001F24: 916A0D7E
	s_mov_b32 s45, 0                                           // 000000001F28: BEAD0080
	s_cbranch_vccnz 65411                                      // 000000001F2C: BFA4FF83 <r_2_4_14_16_4_3_3+0x53c>
	s_add_i32 s45, s33, s1                                     // 000000001F30: 812D0121
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F34: BF870499
	s_mul_hi_i32 s46, s45, 0x2aaaaaab                          // 000000001F38: 972EFF2D 2AAAAAAB
	s_lshr_b32 s47, s46, 31                                    // 000000001F40: 852F9F2E
	s_ashr_i32 s46, s46, 3                                     // 000000001F44: 862E832E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F48: BF870499
	s_add_i32 s46, s46, s47                                    // 000000001F4C: 812E2F2E
	s_mul_i32 s46, s46, 48                                     // 000000001F50: 962EB02E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F54: BF870499
	s_sub_i32 s45, s45, s46                                    // 000000001F58: 81AD2E2D
	s_mul_i32 s46, s45, 5                                      // 000000001F5C: 962E852D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F60: BF870499
	s_ashr_i32 s47, s46, 31                                    // 000000001F64: 862F9F2E
	s_lshl_b64 s[46:47], s[46:47], 2                           // 000000001F68: 84AE822E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F6C: BF870009
	s_add_u32 s46, s25, s46                                    // 000000001F70: 802E2E19
	s_addc_u32 s47, s26, s47                                   // 000000001F74: 822F2F1A
	s_load_b32 s45, s[46:47], null                             // 000000001F78: F4000B57 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s29                        // 000000001F80: 916A1D7E
	s_cbranch_vccnz 65391                                      // 000000001F84: BFA4FF6F <r_2_4_14_16_4_3_3+0x544>
	s_add_i32 s44, s31, s1                                     // 000000001F88: 812C011F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F8C: BF870499
	s_mul_hi_i32 s46, s44, 0x2aaaaaab                          // 000000001F90: 972EFF2C 2AAAAAAB
	s_lshr_b32 s47, s46, 31                                    // 000000001F98: 852F9F2E
	s_ashr_i32 s46, s46, 3                                     // 000000001F9C: 862E832E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001FA0: BF870499
	s_add_i32 s46, s46, s47                                    // 000000001FA4: 812E2F2E
	s_mul_i32 s46, s46, 48                                     // 000000001FA8: 962EB02E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001FAC: BF870499
	s_sub_i32 s44, s44, s46                                    // 000000001FB0: 81AC2E2C
	s_mul_i32 s46, s44, 5                                      // 000000001FB4: 962E852C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001FB8: BF870499
	s_ashr_i32 s47, s46, 31                                    // 000000001FBC: 862F9F2E
	s_lshl_b64 s[46:47], s[46:47], 2                           // 000000001FC0: 84AE822E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001FC4: BF870009
	s_add_u32 s46, s18, s46                                    // 000000001FC8: 802E2E12
	s_addc_u32 s47, s19, s47                                   // 000000001FCC: 822F2F13
	s_load_b32 s44, s[46:47], null                             // 000000001FD0: F4000B17 F8000000
	s_mov_b32 s46, 0                                           // 000000001FD8: BEAE0080
	s_and_not1_b32 vcc_lo, exec_lo, s11                        // 000000001FDC: 916A0B7E
	s_mov_b32 s47, 0                                           // 000000001FE0: BEAF0080
	s_cbranch_vccnz 65371                                      // 000000001FE4: BFA4FF5B <r_2_4_14_16_4_3_3+0x554>
	s_add_i32 s47, s30, s1                                     // 000000001FE8: 812F011E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001FEC: BF870499
	s_mul_hi_i32 s48, s47, 0x2aaaaaab                          // 000000001FF0: 9730FF2F 2AAAAAAB
	s_lshr_b32 s49, s48, 31                                    // 000000001FF8: 85319F30
	s_ashr_i32 s48, s48, 3                                     // 000000001FFC: 86308330
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002000: BF870499
	s_add_i32 s48, s48, s49                                    // 000000002004: 81303130
	s_mul_i32 s48, s48, 48                                     // 000000002008: 9630B030
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000200C: BF870499
	s_sub_i32 s47, s47, s48                                    // 000000002010: 81AF302F
	s_mul_i32 s48, s47, 5                                      // 000000002014: 9630852F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002018: BF870499
	s_ashr_i32 s49, s48, 31                                    // 00000000201C: 86319F30
	s_lshl_b64 s[48:49], s[48:49], 2                           // 000000002020: 84B08230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002024: BF870009
	s_add_u32 s48, s6, s48                                     // 000000002028: 80303006
	s_addc_u32 s49, s7, s49                                    // 00000000202C: 82313107
	s_load_b32 s47, s[48:49], null                             // 000000002030: F4000BD8 F8000000
	s_and_not1_b32 vcc_lo, exec_lo, s36                        // 000000002038: 916A247E
	s_cbranch_vccnz 65290                                      // 00000000203C: BFA4FF0A <r_2_4_14_16_4_3_3+0x468>
	s_add_i32 s46, s17, s1                                     // 000000002040: 812E0111
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002044: BF870499
	s_mul_hi_i32 s48, s46, 0x2aaaaaab                          // 000000002048: 9730FF2E 2AAAAAAB
	s_lshr_b32 s49, s48, 31                                    // 000000002050: 85319F30
	s_ashr_i32 s48, s48, 3                                     // 000000002054: 86308330
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002058: BF870499
	s_add_i32 s48, s48, s49                                    // 00000000205C: 81303130
	s_mul_i32 s48, s48, 48                                     // 000000002060: 9630B030
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002064: BF870499
	s_sub_i32 s46, s46, s48                                    // 000000002068: 81AE302E
	s_mul_i32 s48, s46, 5                                      // 00000000206C: 9630852E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002070: BF870499
	s_ashr_i32 s49, s48, 31                                    // 000000002074: 86319F30
	s_lshl_b64 s[48:49], s[48:49], 2                           // 000000002078: 84B08230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000207C: BF870009
	s_add_u32 s48, s25, s48                                    // 000000002080: 80303019
	s_addc_u32 s49, s26, s49                                   // 000000002084: 8231311A
	s_load_b32 s46, s[48:49], null                             // 000000002088: F4000B98 F8000000
	s_branch 65269                                             // 000000002090: BFA0FEF5 <r_2_4_14_16_4_3_3+0x468>
	s_mul_i32 s0, s2, 0x380                                    // 000000002094: 9600FF02 00000380
	s_mul_i32 s6, s14, 0xe0                                    // 00000000209C: 9606FF0E 000000E0
	s_ashr_i32 s1, s0, 31                                      // 0000000020A4: 86019F00
	v_dual_add_f32 v0, s10, v0 :: v_dual_mov_b32 v1, 0         // 0000000020A8: C910000A 00000080
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000020B0: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000020B4: BF8704D9
	s_add_u32 s2, s4, s0                                       // 0000000020B8: 80020004
	s_addc_u32 s4, s5, s1                                      // 0000000020BC: 82040105
	s_ashr_i32 s7, s6, 31                                      // 0000000020C0: 86079F06
	v_max_f32_e32 v0, 0, v0                                    // 0000000020C4: 20000080
	s_lshl_b64 s[0:1], s[6:7], 2                               // 0000000020C8: 84808206
	s_add_u32 s2, s2, s0                                       // 0000000020CC: 80020002
	s_addc_u32 s4, s4, s1                                      // 0000000020D0: 82040104
	s_lshl_b32 s0, s3, 4                                       // 0000000020D4: 84008403
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000020D8: BF870499
	s_ashr_i32 s1, s0, 31                                      // 0000000020DC: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000020E0: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000020E4: BF8704B9
	s_add_u32 s2, s2, s0                                       // 0000000020E8: 80020002
	s_addc_u32 s3, s4, s1                                      // 0000000020EC: 82030104
	s_ashr_i32 s13, s12, 31                                    // 0000000020F0: 860D9F0C
	s_lshl_b64 s[0:1], s[12:13], 2                             // 0000000020F4: 8480820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000020F8: BF870009
	s_add_u32 s0, s2, s0                                       // 0000000020FC: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000002100: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000002104: DC6A0000 00000001
	s_nop 0                                                    // 00000000210C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000002110: BFB60003
	s_endpgm                                                   // 000000002114: BFB00000
