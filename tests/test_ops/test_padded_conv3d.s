
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_9_9_9_4_3_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b64 s[2:3], s[0:1], 0x10                            // 000000001704: F4040080 F8000010
	s_load_b128 s[24:27], s[0:1], null                         // 00000000170C: F4080600 F8000000
	s_mul_hi_i32 s0, s13, 0x38e38e39                           // 000000001714: 9700FF0D 38E38E39
	v_mov_b32_e32 v0, 0                                        // 00000000171C: 7E000280
	s_lshr_b32 s1, s0, 31                                      // 000000001720: 85019F00
	s_ashr_i32 s4, s0, 1                                       // 000000001724: 86048100
	s_mul_i32 s0, s15, 0x6c                                    // 000000001728: 9600FF0F 0000006C
	s_add_i32 s8, s4, s1                                       // 000000001730: 81080104
	s_ashr_i32 s1, s0, 31                                      // 000000001734: 86019F00
	s_mul_i32 s4, s8, 9                                        // 000000001738: 96048908
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000173C: 84808200
	s_sub_i32 s6, s13, s4                                      // 000000001740: 8186040D
	s_waitcnt lgkmcnt(0)                                       // 000000001744: BF89FC07
	s_add_u32 s9, s2, s0                                       // 000000001748: 80090002
	s_addc_u32 s33, s3, s1                                     // 00000000174C: 82210103
	s_ashr_i32 s5, s4, 31                                      // 000000001750: 86059F04
	s_mul_i32 s0, s14, 0x51                                    // 000000001754: 9600FF0E 00000051
	s_lshl_b64 s[10:11], s[4:5], 2                             // 00000000175C: 848A8204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001760: BF870009
	s_add_u32 s2, s26, s10                                     // 000000001764: 80020A1A
	s_addc_u32 s3, s27, s11                                    // 000000001768: 82030B1B
	s_add_i32 s4, s14, -1                                      // 00000000176C: 8104C10E
	s_ashr_i32 s1, s0, 31                                      // 000000001770: 86019F00
	s_cmp_gt_i32 s4, -1                                        // 000000001774: BF02C104
	s_cselect_b32 s4, -1, 0                                    // 000000001778: 980480C1
	s_cmp_lt_i32 s14, 10                                       // 00000000177C: BF048A0E
	s_cselect_b32 s5, -1, 0                                    // 000000001780: 980580C1
	s_add_i32 s12, s8, -1                                      // 000000001784: 810CC108
	s_cmpk_lt_i32 s13, 0x5a                                    // 000000001788: B38D005A
	s_cselect_b32 s16, -1, 0                                   // 00000000178C: 981080C1
	s_add_i32 s17, s6, -1                                      // 000000001790: 8111C106
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001794: BF870499
	s_or_b32 s7, s17, s12                                      // 000000001798: 8C070C11
	s_cmp_gt_i32 s7, -1                                        // 00000000179C: BF02C107
	s_cselect_b32 s7, -1, 0                                    // 0000000017A0: 980780C1
	s_lshl_b64 s[26:27], s[0:1], 2                             // 0000000017A4: 849A8200
	s_and_b32 s18, s16, s7                                     // 0000000017A8: 8B120710
	s_ashr_i32 s7, s6, 31                                      // 0000000017AC: 86079F06
	s_and_b32 s19, s4, s18                                     // 0000000017B0: 8B131204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000017B4: BF8704C9
	s_and_b32 s19, s5, s19                                     // 0000000017B8: 8B131305
	s_add_u32 s0, s2, s26                                      // 0000000017BC: 80001A02
	s_addc_u32 s1, s3, s27                                     // 0000000017C0: 82011B03
	s_or_b32 s2, s6, s12                                       // 0000000017C4: 8C020C06
	s_cmp_gt_i32 s2, -1                                        // 0000000017C8: BF02C102
	s_cselect_b32 s2, -1, 0                                    // 0000000017CC: 980280C1
	s_add_i32 s20, s6, 1                                       // 0000000017D0: 81148106
	s_and_b32 s2, s16, s2                                      // 0000000017D4: 8B020210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017D8: BF870499
	s_and_b32 s3, s4, s2                                       // 0000000017DC: 8B030204
	s_and_b32 s3, s5, s3                                       // 0000000017E0: 8B030305
	s_cmp_lt_i32 s6, 8                                         // 0000000017E4: BF048806
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 0000000017E8: D5010001 000D0280
	s_cselect_b32 s21, -1, 0                                   // 0000000017F0: 981580C1
	s_or_b32 s12, s20, s12                                     // 0000000017F4: 8C0C0C14
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017F8: BF8704A9
	s_cmp_gt_i32 s12, -1                                       // 0000000017FC: BF02C10C
	s_cselect_b32 s12, -1, 0                                   // 000000001800: 980C80C1
	s_and_b32 s12, s21, s12                                    // 000000001804: 8B0C0C15
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001808: BF870499
	s_and_b32 s16, s16, s12                                    // 00000000180C: 8B100C10
	s_and_b32 s12, s4, s16                                     // 000000001810: 8B0C1004
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001814: BF8704D9
	s_and_b32 s22, s5, s12                                     // 000000001818: 8B160C05
	s_cmpk_lt_i32 s13, 0x51                                    // 00000000181C: B38D0051
	v_cndmask_b32_e64 v2, 0, 1, s22                            // 000000001820: D5010002 00590280
	s_cselect_b32 s12, -1, 0                                   // 000000001828: 980C80C1
	s_or_b32 s23, s17, s8                                      // 00000000182C: 8C170811
	s_cmp_gt_i32 s23, -1                                       // 000000001830: BF02C117
	s_cselect_b32 s23, -1, 0                                   // 000000001834: 981780C1
	s_or_b32 s29, s6, s8                                       // 000000001838: 8C1D0806
	s_and_b32 s23, s12, s23                                    // 00000000183C: 8B17170C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001840: BF870499
	s_and_b32 s28, s4, s23                                     // 000000001844: 8B1C1704
	s_and_b32 s28, s5, s28                                     // 000000001848: 8B1C1C05
	s_cmp_gt_i32 s29, -1                                       // 00000000184C: BF02C11D
	v_cndmask_b32_e64 v3, 0, 1, s28                            // 000000001850: D5010003 00710280
	s_cselect_b32 s29, -1, 0                                   // 000000001858: 981D80C1
	s_or_b32 s31, s20, s8                                      // 00000000185C: 8C1F0814
	s_and_b32 s29, s12, s29                                    // 000000001860: 8B1D1D0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001864: BF870499
	s_and_b32 s30, s4, s29                                     // 000000001868: 8B1E1D04
	s_and_b32 s30, s5, s30                                     // 00000000186C: 8B1E1E05
	s_cmp_gt_i32 s31, -1                                       // 000000001870: BF02C11F
	v_cndmask_b32_e64 v4, 0, 1, s30                            // 000000001874: D5010004 00790280
	s_cselect_b32 s31, -1, 0                                   // 00000000187C: 981F80C1
	s_add_i32 s8, s8, 1                                        // 000000001880: 81088108
	s_and_b32 s31, s21, s31                                    // 000000001884: 8B1F1F15
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001888: BF870499
	s_and_b32 s31, s12, s31                                    // 00000000188C: 8B1F1F0C
	s_and_b32 s12, s4, s31                                     // 000000001890: 8B0C1F04
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001894: BF8704D9
	s_and_b32 s34, s5, s12                                     // 000000001898: 8B220C05
	s_cmpk_lt_i32 s13, 0x48                                    // 00000000189C: B38D0048
	v_cndmask_b32_e64 v5, 0, 1, s34                            // 0000000018A0: D5010005 00890280
	s_cselect_b32 s35, -1, 0                                   // 0000000018A8: 982380C1
	s_or_b32 s12, s17, s8                                      // 0000000018AC: 8C0C0811
	s_cmp_gt_i32 s12, -1                                       // 0000000018B0: BF02C10C
	s_cselect_b32 s12, -1, 0                                   // 0000000018B4: 980C80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000018B8: BF8704B9
	s_and_b32 s17, s35, s12                                    // 0000000018BC: 8B110C23
	s_lshl_b64 s[12:13], s[6:7], 2                             // 0000000018C0: 848C8206
	s_and_b32 s36, s4, s17                                     // 0000000018C4: 8B241104
	s_and_b32 s7, s5, s36                                      // 0000000018C8: 8B072405
	s_add_u32 s0, s0, s12                                      // 0000000018CC: 80000C00
	s_addc_u32 s36, s1, s13                                    // 0000000018D0: 82240D01
	s_or_b32 s1, s6, s8                                        // 0000000018D4: 8C010806
	v_cndmask_b32_e64 v6, 0, 1, s7                             // 0000000018D8: D5010006 001D0280
	s_cmp_gt_i32 s1, -1                                        // 0000000018E0: BF02C101
	s_cselect_b32 s1, -1, 0                                    // 0000000018E4: 980180C1
	s_or_b32 s6, s20, s8                                       // 0000000018E8: 8C060814
	s_and_b32 s1, s35, s1                                      // 0000000018EC: 8B010123
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018F0: BF870499
	s_and_b32 s3, s4, s1                                       // 0000000018F4: 8B030104
	s_and_b32 s3, s5, s3                                       // 0000000018F8: 8B030305
	s_cmp_gt_i32 s6, -1                                        // 0000000018FC: BF02C106
	v_cndmask_b32_e64 v7, 0, 1, s3                             // 000000001900: D5010007 000D0280
	s_cselect_b32 s6, -1, 0                                    // 000000001908: 980680C1
	s_add_i32 s8, s14, 1                                       // 00000000190C: 8108810E
	s_and_b32 s6, s21, s6                                      // 000000001910: 8B060615
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001914: BF870499
	s_and_b32 s6, s35, s6                                      // 000000001918: 8B060623
	s_and_b32 s4, s4, s6                                       // 00000000191C: 8B040604
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001920: BF870009
	s_and_b32 s4, s5, s4                                       // 000000001924: 8B040405
	s_cmp_gt_i32 s14, -1                                       // 000000001928: BF02C10E
	v_cndmask_b32_e64 v8, 0, 1, s4                             // 00000000192C: D5010008 00110280
	s_cselect_b32 s5, -1, 0                                    // 000000001934: 980580C1
	s_cmp_lt_i32 s8, 10                                        // 000000001938: BF048A08
	s_cselect_b32 s3, -1, 0                                    // 00000000193C: 980380C1
	s_and_b32 s7, s5, s18                                      // 000000001940: 8B071205
	s_and_b32 s20, s5, s2                                      // 000000001944: 8B140205
	s_and_b32 s21, s5, s16                                     // 000000001948: 8B151005
	s_and_b32 s22, s5, s23                                     // 00000000194C: 8B161705
	s_and_b32 s28, s5, s29                                     // 000000001950: 8B1C1D05
	s_and_b32 s30, s5, s31                                     // 000000001954: 8B1E1F05
	s_and_b32 s34, s5, s17                                     // 000000001958: 8B221105
	s_and_b32 s35, s5, s1                                      // 00000000195C: 8B230105
	s_and_b32 s5, s5, s6                                       // 000000001960: 8B050605
	s_add_i32 s37, s14, 2                                      // 000000001964: 8125820E
	s_and_b32 s14, s3, s7                                      // 000000001968: 8B0E0703
	s_and_b32 s38, s3, s20                                     // 00000000196C: 8B261403
	s_and_b32 s39, s3, s21                                     // 000000001970: 8B271503
	s_and_b32 s40, s3, s22                                     // 000000001974: 8B281603
	s_and_b32 s41, s3, s28                                     // 000000001978: 8B291C03
	s_and_b32 s42, s3, s30                                     // 00000000197C: 8B2A1E03
	s_and_b32 s43, s3, s34                                     // 000000001980: 8B2B2203
	s_and_b32 s44, s3, s35                                     // 000000001984: 8B2C2303
	s_and_b32 s45, s3, s5                                      // 000000001988: 8B2D0503
	s_cmp_gt_i32 s8, -1                                        // 00000000198C: BF02C108
	s_cselect_b32 s3, -1, 0                                    // 000000001990: 980380C1
	s_cmp_lt_i32 s37, 10                                       // 000000001994: BF048A25
	s_cselect_b32 s4, -1, 0                                    // 000000001998: 980480C1
	s_and_b32 s5, s3, s18                                      // 00000000199C: 8B051203
	s_and_b32 s2, s3, s2                                       // 0000000019A0: 8B020203
	s_and_b32 s7, s3, s16                                      // 0000000019A4: 8B071003
	s_and_b32 s8, s3, s23                                      // 0000000019A8: 8B081703
	s_and_b32 s16, s3, s29                                     // 0000000019AC: 8B101D03
	s_and_b32 s18, s3, s31                                     // 0000000019B0: 8B121F03
	s_and_b32 s17, s3, s17                                     // 0000000019B4: 8B111103
	s_and_b32 s1, s3, s1                                       // 0000000019B8: 8B010103
	s_and_b32 s3, s3, s6                                       // 0000000019BC: 8B030603
	s_and_b32 s46, s4, s5                                      // 0000000019C0: 8B2E0504
	s_and_b32 s47, s4, s2                                      // 0000000019C4: 8B2F0204
	s_and_b32 s48, s4, s7                                      // 0000000019C8: 8B300704
	s_and_b32 s49, s4, s8                                      // 0000000019CC: 8B310804
	s_and_b32 s50, s4, s16                                     // 0000000019D0: 8B321004
	s_and_b32 s51, s4, s18                                     // 0000000019D4: 8B331204
	s_and_b32 s52, s4, s17                                     // 0000000019D8: 8B341104
	s_and_b32 s53, s4, s1                                      // 0000000019DC: 8B350104
	s_and_b32 s54, s4, s3                                      // 0000000019E0: 8B360304
	s_add_u32 s28, s0, 0x16c                                   // 0000000019E4: 801CFF00 0000016C
	v_cmp_ne_u32_e64 s0, 1, v1                                 // 0000000019EC: D44D0000 00020281
	v_cmp_ne_u32_e64 s1, 1, v2                                 // 0000000019F4: D44D0001 00020481
	v_cmp_ne_u32_e64 s2, 1, v3                                 // 0000000019FC: D44D0002 00020681
	v_cmp_ne_u32_e64 s3, 1, v4                                 // 000000001A04: D44D0003 00020881
	v_cmp_ne_u32_e64 s4, 1, v5                                 // 000000001A0C: D44D0004 00020A81
	v_cmp_ne_u32_e64 s5, 1, v6                                 // 000000001A14: D44D0005 00020C81
	v_cmp_ne_u32_e64 s6, 1, v7                                 // 000000001A1C: D44D0006 00020E81
	v_cmp_ne_u32_e64 s7, 1, v8                                 // 000000001A24: D44D0007 00021081
	s_addc_u32 s29, s36, 0                                     // 000000001A2C: 821D8024
	s_mov_b64 s[30:31], 0                                      // 000000001A30: BE9E0180
	s_and_b32 s8, exec_lo, s19                                 // 000000001A34: 8B08137E
	s_branch 79                                                // 000000001A38: BFA0004F <r_4_9_9_9_4_3_3_3+0x478>
	s_waitcnt lgkmcnt(0)                                       // 000000001A3C: BF89FC07
	v_fmac_f32_e64 v0, s55, s16                                // 000000001A40: D52B0000 00002037
	s_load_b32 s16, s[34:35], 0x68                             // 000000001A48: F4000411 F8000068
	s_add_u32 s30, s30, 0x6c                                   // 000000001A50: 801EFF1E 0000006C
	s_addc_u32 s31, s31, 0                                     // 000000001A58: 821F801F
	s_add_u32 s28, s28, 0xb64                                  // 000000001A5C: 801CFF1C 00000B64
	v_fmac_f32_e64 v0, s57, s17                                // 000000001A64: D52B0000 00002239
	s_addc_u32 s29, s29, 0                                     // 000000001A6C: 821D801D
	s_cmpk_eq_i32 s30, 0x1b0                                   // 000000001A70: B19E01B0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A74: BF870091
	v_fmac_f32_e64 v0, s56, s18                                // 000000001A78: D52B0000 00002438
	v_fmac_f32_e64 v0, s59, s19                                // 000000001A80: D52B0000 0000263B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A88: BF870091
	v_fmac_f32_e64 v0, s58, s20                                // 000000001A8C: D52B0000 0000283A
	v_fmac_f32_e64 v0, s61, s21                                // 000000001A94: D52B0000 00002A3D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A9C: BF870091
	v_fmac_f32_e64 v0, s60, s22                                // 000000001AA0: D52B0000 00002C3C
	v_fmac_f32_e64 v0, s63, s23                                // 000000001AA8: D52B0000 00002E3F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AB0: BF870091
	v_fmac_f32_e64 v0, s62, s36                                // 000000001AB4: D52B0000 0000483E
	v_fmac_f32_e64 v0, s65, s37                                // 000000001ABC: D52B0000 00004A41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AC4: BF870091
	v_fmac_f32_e64 v0, s64, s67                                // 000000001AC8: D52B0000 00008640
	v_fmac_f32_e64 v0, s68, s69                                // 000000001AD0: D52B0000 00008A44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AD8: BF870091
	v_fmac_f32_e64 v0, s66, s71                                // 000000001ADC: D52B0000 00008E42
	v_fmac_f32_e64 v0, s72, s73                                // 000000001AE4: D52B0000 00009248
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AEC: BF870091
	v_fmac_f32_e64 v0, s70, s75                                // 000000001AF0: D52B0000 00009646
	v_fmac_f32_e64 v0, s76, s77                                // 000000001AF8: D52B0000 00009A4C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B00: BF870091
	v_fmac_f32_e64 v0, s74, s79                                // 000000001B04: D52B0000 00009E4A
	v_fmac_f32_e64 v0, s80, s81                                // 000000001B0C: D52B0000 0000A250
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B14: BF870091
	v_fmac_f32_e64 v0, s78, s83                                // 000000001B18: D52B0000 0000A64E
	v_fmac_f32_e64 v0, s84, s85                                // 000000001B20: D52B0000 0000AA54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B28: BF870091
	v_fmac_f32_e64 v0, s82, s87                                // 000000001B2C: D52B0000 0000AE52
	v_fmac_f32_e64 v0, s88, s89                                // 000000001B34: D52B0000 0000B258
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B3C: BF870091
	v_fmac_f32_e64 v0, s86, s90                                // 000000001B40: D52B0000 0000B456
	v_fmac_f32_e64 v0, s92, s93                                // 000000001B48: D52B0000 0000BA5C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B50: BF870091
	v_fmac_f32_e64 v0, s91, s95                                // 000000001B54: D52B0000 0000BE5B
	v_fmac_f32_e64 v0, s96, s97                                // 000000001B5C: D52B0000 0000C260
	s_waitcnt lgkmcnt(0)                                       // 000000001B64: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B68: BF870001
	v_fmac_f32_e64 v0, s94, s16                                // 000000001B6C: D52B0000 0000205E
	s_cbranch_scc1 207                                         // 000000001B74: BFA200CF <r_4_9_9_9_4_3_3_3+0x7b4>
	s_mov_b32 s55, 0                                           // 000000001B78: BEB70080
	s_mov_b32 vcc_lo, s8                                       // 000000001B7C: BEEA0008
	s_cbranch_vccz 2                                           // 000000001B80: BFA30002 <r_4_9_9_9_4_3_3_3+0x48c>
	s_load_b32 s55, s[28:29], -0x2d8                           // 000000001B84: F4000DCE F81FFD28
	s_add_u32 s34, s9, s30                                     // 000000001B8C: 80221E09
	s_addc_u32 s35, s33, s31                                   // 000000001B90: 82231F21
	s_mov_b32 s56, 0                                           // 000000001B94: BEB80080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001B98: 8B6A007E
	s_mov_b32 s57, 0                                           // 000000001B9C: BEB90080
	s_cbranch_vccz 146                                         // 000000001BA0: BFA30092 <r_4_9_9_9_4_3_3_3+0x6ec>
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001BA4: 8B6A017E
	s_cbranch_vccz 148                                         // 000000001BA8: BFA30094 <r_4_9_9_9_4_3_3_3+0x6fc>
	s_mov_b32 s58, 0                                           // 000000001BAC: BEBA0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001BB0: 8B6A027E
	s_mov_b32 s59, 0                                           // 000000001BB4: BEBB0080
	s_cbranch_vccz 150                                         // 000000001BB8: BFA30096 <r_4_9_9_9_4_3_3_3+0x714>
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001BBC: 8B6A037E
	s_cbranch_vccz 152                                         // 000000001BC0: BFA30098 <r_4_9_9_9_4_3_3_3+0x724>
	s_mov_b32 s60, 0                                           // 000000001BC4: BEBC0080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001BC8: 8B6A047E
	s_mov_b32 s61, 0                                           // 000000001BCC: BEBD0080
	s_cbranch_vccz 154                                         // 000000001BD0: BFA3009A <r_4_9_9_9_4_3_3_3+0x73c>
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001BD4: 8B6A057E
	s_cbranch_vccz 156                                         // 000000001BD8: BFA3009C <r_4_9_9_9_4_3_3_3+0x74c>
	s_mov_b32 s62, 0                                           // 000000001BDC: BEBE0080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001BE0: 8B6A067E
	s_mov_b32 s63, 0                                           // 000000001BE4: BEBF0080
	s_cbranch_vccz 158                                         // 000000001BE8: BFA3009E <r_4_9_9_9_4_3_3_3+0x764>
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001BEC: 8B6A077E
	s_cbranch_vccz 160                                         // 000000001BF0: BFA300A0 <r_4_9_9_9_4_3_3_3+0x774>
	s_mov_b32 s64, 0                                           // 000000001BF4: BEC00080
	s_and_not1_b32 vcc_lo, exec_lo, s14                        // 000000001BF8: 916A0E7E
	s_mov_b32 s65, 0                                           // 000000001BFC: BEC10080
	s_cbranch_vccz 162                                         // 000000001C00: BFA300A2 <r_4_9_9_9_4_3_3_3+0x78c>
	s_clause 0x1                                               // 000000001C04: BF850001
	s_load_b256 s[16:23], s[34:35], null                       // 000000001C08: F40C0411 F8000000
	s_load_b64 s[36:37], s[34:35], 0x20                        // 000000001C10: F4040911 F8000020
	s_and_not1_b32 vcc_lo, exec_lo, s38                        // 000000001C18: 916A267E
	s_cbranch_vccnz 2                                          // 000000001C1C: BFA40002 <r_4_9_9_9_4_3_3_3+0x528>
	s_load_b32 s64, s[28:29], -0x190                           // 000000001C20: F400100E F81FFE70
	s_load_b32 s67, s[34:35], 0x28                             // 000000001C28: F40010D1 F8000028
	s_mov_b32 s66, 0                                           // 000000001C30: BEC20080
	s_and_not1_b32 vcc_lo, exec_lo, s39                        // 000000001C34: 916A277E
	s_mov_b32 s68, 0                                           // 000000001C38: BEC40080
	s_cbranch_vccnz 2                                          // 000000001C3C: BFA40002 <r_4_9_9_9_4_3_3_3+0x548>
	s_load_b32 s68, s[28:29], -0x18c                           // 000000001C40: F400110E F81FFE74
	s_load_b32 s69, s[34:35], 0x2c                             // 000000001C48: F4001151 F800002C
	s_and_not1_b32 vcc_lo, exec_lo, s40                        // 000000001C50: 916A287E
	s_cbranch_vccnz 2                                          // 000000001C54: BFA40002 <r_4_9_9_9_4_3_3_3+0x560>
	s_load_b32 s66, s[28:29], -0x170                           // 000000001C58: F400108E F81FFE90
	s_load_b32 s71, s[34:35], 0x30                             // 000000001C60: F40011D1 F8000030
	s_mov_b32 s70, 0                                           // 000000001C68: BEC60080
	s_and_not1_b32 vcc_lo, exec_lo, s41                        // 000000001C6C: 916A297E
	s_mov_b32 s72, 0                                           // 000000001C70: BEC80080
	s_cbranch_vccnz 2                                          // 000000001C74: BFA40002 <r_4_9_9_9_4_3_3_3+0x580>
	s_load_b32 s72, s[28:29], -0x16c                           // 000000001C78: F400120E F81FFE94
	s_load_b32 s73, s[34:35], 0x34                             // 000000001C80: F4001251 F8000034
	s_and_not1_b32 vcc_lo, exec_lo, s42                        // 000000001C88: 916A2A7E
	s_cbranch_vccnz 2                                          // 000000001C8C: BFA40002 <r_4_9_9_9_4_3_3_3+0x598>
	s_load_b32 s70, s[28:29], -0x168                           // 000000001C90: F400118E F81FFE98
	s_load_b32 s75, s[34:35], 0x38                             // 000000001C98: F40012D1 F8000038
	s_mov_b32 s74, 0                                           // 000000001CA0: BECA0080
	s_and_not1_b32 vcc_lo, exec_lo, s43                        // 000000001CA4: 916A2B7E
	s_mov_b32 s76, 0                                           // 000000001CA8: BECC0080
	s_cbranch_vccnz 2                                          // 000000001CAC: BFA40002 <r_4_9_9_9_4_3_3_3+0x5b8>
	s_load_b32 s76, s[28:29], -0x14c                           // 000000001CB0: F400130E F81FFEB4
	s_load_b32 s77, s[34:35], 0x3c                             // 000000001CB8: F4001351 F800003C
	s_and_not1_b32 vcc_lo, exec_lo, s44                        // 000000001CC0: 916A2C7E
	s_cbranch_vccnz 2                                          // 000000001CC4: BFA40002 <r_4_9_9_9_4_3_3_3+0x5d0>
	s_load_b32 s74, s[28:29], -0x148                           // 000000001CC8: F400128E F81FFEB8
	s_load_b32 s79, s[34:35], 0x40                             // 000000001CD0: F40013D1 F8000040
	s_mov_b32 s78, 0                                           // 000000001CD8: BECE0080
	s_and_not1_b32 vcc_lo, exec_lo, s45                        // 000000001CDC: 916A2D7E
	s_mov_b32 s80, 0                                           // 000000001CE0: BED00080
	s_cbranch_vccnz 2                                          // 000000001CE4: BFA40002 <r_4_9_9_9_4_3_3_3+0x5f0>
	s_load_b32 s80, s[28:29], -0x144                           // 000000001CE8: F400140E F81FFEBC
	s_load_b32 s81, s[34:35], 0x44                             // 000000001CF0: F4001451 F8000044
	s_and_not1_b32 vcc_lo, exec_lo, s46                        // 000000001CF8: 916A2E7E
	s_cbranch_vccnz 2                                          // 000000001CFC: BFA40002 <r_4_9_9_9_4_3_3_3+0x608>
	s_load_b32 s78, s[28:29], -0x50                            // 000000001D00: F400138E F81FFFB0
	s_load_b32 s83, s[34:35], 0x48                             // 000000001D08: F40014D1 F8000048
	s_mov_b32 s82, 0                                           // 000000001D10: BED20080
	s_and_not1_b32 vcc_lo, exec_lo, s47                        // 000000001D14: 916A2F7E
	s_mov_b32 s84, 0                                           // 000000001D18: BED40080
	s_cbranch_vccnz 2                                          // 000000001D1C: BFA40002 <r_4_9_9_9_4_3_3_3+0x628>
	s_load_b32 s84, s[28:29], -0x4c                            // 000000001D20: F400150E F81FFFB4
	s_load_b32 s85, s[34:35], 0x4c                             // 000000001D28: F4001551 F800004C
	s_and_not1_b32 vcc_lo, exec_lo, s48                        // 000000001D30: 916A307E
	s_cbranch_vccnz 2                                          // 000000001D34: BFA40002 <r_4_9_9_9_4_3_3_3+0x640>
	s_load_b32 s82, s[28:29], -0x48                            // 000000001D38: F400148E F81FFFB8
	s_load_b32 s87, s[34:35], 0x50                             // 000000001D40: F40015D1 F8000050
	s_mov_b32 s86, 0                                           // 000000001D48: BED60080
	s_and_not1_b32 vcc_lo, exec_lo, s49                        // 000000001D4C: 916A317E
	s_mov_b32 s88, 0                                           // 000000001D50: BED80080
	s_cbranch_vccnz 2                                          // 000000001D54: BFA40002 <r_4_9_9_9_4_3_3_3+0x660>
	s_load_b32 s88, s[28:29], -0x2c                            // 000000001D58: F400160E F81FFFD4
	s_load_b32 s89, s[34:35], 0x54                             // 000000001D60: F4001651 F8000054
	s_and_not1_b32 vcc_lo, exec_lo, s50                        // 000000001D68: 916A327E
	s_cbranch_vccnz 2                                          // 000000001D6C: BFA40002 <r_4_9_9_9_4_3_3_3+0x678>
	s_load_b32 s86, s[28:29], -0x28                            // 000000001D70: F400158E F81FFFD8
	s_load_b32 s90, s[34:35], 0x58                             // 000000001D78: F4001691 F8000058
	s_mov_b32 s91, 0                                           // 000000001D80: BEDB0080
	s_and_not1_b32 vcc_lo, exec_lo, s51                        // 000000001D84: 916A337E
	s_mov_b32 s92, 0                                           // 000000001D88: BEDC0080
	s_cbranch_vccnz 2                                          // 000000001D8C: BFA40002 <r_4_9_9_9_4_3_3_3+0x698>
	s_load_b32 s92, s[28:29], -0x24                            // 000000001D90: F400170E F81FFFDC
	s_load_b32 s93, s[34:35], 0x5c                             // 000000001D98: F4001751 F800005C
	s_and_not1_b32 vcc_lo, exec_lo, s52                        // 000000001DA0: 916A347E
	s_cbranch_vccnz 2                                          // 000000001DA4: BFA40002 <r_4_9_9_9_4_3_3_3+0x6b0>
	s_load_b32 s91, s[28:29], -0x8                             // 000000001DA8: F40016CE F81FFFF8
	s_load_b32 s95, s[34:35], 0x60                             // 000000001DB0: F40017D1 F8000060
	s_mov_b32 s94, 0                                           // 000000001DB8: BEDE0080
	s_and_not1_b32 vcc_lo, exec_lo, s53                        // 000000001DBC: 916A357E
	s_mov_b32 s96, 0                                           // 000000001DC0: BEE00080
	s_cbranch_vccnz 2                                          // 000000001DC4: BFA40002 <r_4_9_9_9_4_3_3_3+0x6d0>
	s_load_b32 s96, s[28:29], -0x4                             // 000000001DC8: F400180E F81FFFFC
	s_load_b32 s97, s[34:35], 0x64                             // 000000001DD0: F4001851 F8000064
	s_and_not1_b32 vcc_lo, exec_lo, s54                        // 000000001DD8: 916A367E
	s_cbranch_vccnz 65303                                      // 000000001DDC: BFA4FF17 <r_4_9_9_9_4_3_3_3+0x33c>
	s_load_b32 s94, s[28:29], null                             // 000000001DE0: F400178E F8000000
	s_branch 65300                                             // 000000001DE8: BFA0FF14 <r_4_9_9_9_4_3_3_3+0x33c>
	s_load_b32 s57, s[28:29], -0x2d4                           // 000000001DEC: F4000E4E F81FFD2C
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001DF4: 8B6A017E
	s_cbranch_vccnz 65388                                      // 000000001DF8: BFA4FF6C <r_4_9_9_9_4_3_3_3+0x4ac>
	s_load_b32 s56, s[28:29], -0x2d0                           // 000000001DFC: F4000E0E F81FFD30
	s_mov_b32 s58, 0                                           // 000000001E04: BEBA0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001E08: 8B6A027E
	s_mov_b32 s59, 0                                           // 000000001E0C: BEBB0080
	s_cbranch_vccnz 65386                                      // 000000001E10: BFA4FF6A <r_4_9_9_9_4_3_3_3+0x4bc>
	s_load_b32 s59, s[28:29], -0x2b4                           // 000000001E14: F4000ECE F81FFD4C
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001E1C: 8B6A037E
	s_cbranch_vccnz 65384                                      // 000000001E20: BFA4FF68 <r_4_9_9_9_4_3_3_3+0x4c4>
	s_load_b32 s58, s[28:29], -0x2b0                           // 000000001E24: F4000E8E F81FFD50
	s_mov_b32 s60, 0                                           // 000000001E2C: BEBC0080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001E30: 8B6A047E
	s_mov_b32 s61, 0                                           // 000000001E34: BEBD0080
	s_cbranch_vccnz 65382                                      // 000000001E38: BFA4FF66 <r_4_9_9_9_4_3_3_3+0x4d4>
	s_load_b32 s61, s[28:29], -0x2ac                           // 000000001E3C: F4000F4E F81FFD54
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001E44: 8B6A057E
	s_cbranch_vccnz 65380                                      // 000000001E48: BFA4FF64 <r_4_9_9_9_4_3_3_3+0x4dc>
	s_load_b32 s60, s[28:29], -0x290                           // 000000001E4C: F4000F0E F81FFD70
	s_mov_b32 s62, 0                                           // 000000001E54: BEBE0080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001E58: 8B6A067E
	s_mov_b32 s63, 0                                           // 000000001E5C: BEBF0080
	s_cbranch_vccnz 65378                                      // 000000001E60: BFA4FF62 <r_4_9_9_9_4_3_3_3+0x4ec>
	s_load_b32 s63, s[28:29], -0x28c                           // 000000001E64: F4000FCE F81FFD74
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001E6C: 8B6A077E
	s_cbranch_vccnz 65376                                      // 000000001E70: BFA4FF60 <r_4_9_9_9_4_3_3_3+0x4f4>
	s_load_b32 s62, s[28:29], -0x288                           // 000000001E74: F4000F8E F81FFD78
	s_mov_b32 s64, 0                                           // 000000001E7C: BEC00080
	s_and_not1_b32 vcc_lo, exec_lo, s14                        // 000000001E80: 916A0E7E
	s_mov_b32 s65, 0                                           // 000000001E84: BEC10080
	s_cbranch_vccnz 65374                                      // 000000001E88: BFA4FF5E <r_4_9_9_9_4_3_3_3+0x504>
	s_load_b32 s65, s[28:29], -0x194                           // 000000001E8C: F400104E F81FFE6C
	s_clause 0x1                                               // 000000001E94: BF850001
	s_load_b256 s[16:23], s[34:35], null                       // 000000001E98: F40C0411 F8000000
	s_load_b64 s[36:37], s[34:35], 0x20                        // 000000001EA0: F4040911 F8000020
	s_and_not1_b32 vcc_lo, exec_lo, s38                        // 000000001EA8: 916A267E
	s_cbranch_vccz 65372                                       // 000000001EAC: BFA3FF5C <r_4_9_9_9_4_3_3_3+0x520>
	s_branch 65373                                             // 000000001EB0: BFA0FF5D <r_4_9_9_9_4_3_3_3+0x528>
	s_mul_i32 s0, s15, 0x2d9                                   // 000000001EB4: 9600FF0F 000002D9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001EBC: BF8704A1
	v_dual_max_f32 v0, v0, v0 :: v_dual_mov_b32 v1, 0          // 000000001EC0: CA900100 00000080
	s_ashr_i32 s1, s0, 31                                      // 000000001EC8: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001ECC: 84808200
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001ED0: BF870001
	v_max_f32_e32 v0, 0, v0                                    // 000000001ED4: 20000080
	s_add_u32 s0, s24, s0                                      // 000000001ED8: 80000018
	s_addc_u32 s1, s25, s1                                     // 000000001EDC: 82010119
	s_add_u32 s0, s0, s26                                      // 000000001EE0: 80001A00
	s_addc_u32 s1, s1, s27                                     // 000000001EE4: 82011B01
	s_add_u32 s0, s0, s10                                      // 000000001EE8: 80000A00
	s_addc_u32 s1, s1, s11                                     // 000000001EEC: 82010B01
	s_add_u32 s0, s0, s12                                      // 000000001EF0: 80000C00
	s_addc_u32 s1, s1, s13                                     // 000000001EF4: 82010D01
	global_store_b32 v1, v0, s[0:1]                            // 000000001EF8: DC6A0000 00000001
	s_nop 0                                                    // 000000001F00: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001F04: BFB60003
	s_endpgm                                                   // 000000001F08: BFB00000
