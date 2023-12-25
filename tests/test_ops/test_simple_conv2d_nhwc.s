
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_20_7_7_10_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[16:19], s[0:1], null                         // 000000001704: F4080400 F8000000
	s_load_b64 s[4:5], s[0:1], 0x10                            // 00000000170C: F4040100 F8000010
	s_mul_hi_i32 s3, s13, 0x92492493                           // 000000001714: 9703FF0D 92492493
	s_mul_i32 s0, s15, 0x32a                                   // 00000000171C: 9600FF0F 0000032A
	s_add_i32 s3, s3, s13                                      // 000000001724: 81030D03
	s_ashr_i32 s1, s0, 31                                      // 000000001728: 86019F00
	s_lshr_b32 s6, s3, 31                                      // 00000000172C: 85069F03
	s_ashr_i32 s3, s3, 2                                       // 000000001730: 86038203
	s_mov_b32 s2, s15                                          // 000000001734: BE82000F
	s_add_i32 s3, s3, s6                                       // 000000001738: 81030603
	s_lshl_b64 s[6:7], s[0:1], 2                               // 00000000173C: 84868200
	s_mul_i32 s12, s3, 7                                       // 000000001740: 960C8703
	s_mul_i32 s8, s3, 0x5a                                     // 000000001744: 9608FF03 0000005A
	s_sub_i32 s0, s13, s12                                     // 00000000174C: 81800C0D
	s_movk_i32 s13, 0x1360                                     // 000000001750: B00D1360
	s_movk_i32 s33, 0x13b0                                     // 000000001754: B02113B0
	s_movk_i32 s56, 0x1400                                     // 000000001758: B0381400
	s_movk_i32 s89, 0x1540                                     // 00000000175C: B0591540
	s_mulk_i32 s2, 0x3d4                                       // 000000001760: B80203D4
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	s_add_u32 s1, s18, s6                                      // 000000001768: 80010612
	s_addc_u32 s3, s19, s7                                     // 00000000176C: 82030713
	s_ashr_i32 s9, s8, 31                                      // 000000001770: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001774: BF870009
	s_lshl_b64 s[6:7], s[8:9], 2                               // 000000001778: 84868208
	s_mul_i32 s8, s0, 10                                       // 00000000177C: 96088A00
	s_add_u32 s1, s1, s6                                       // 000000001780: 80010601
	s_addc_u32 s10, s3, s7                                     // 000000001784: 820A0703
	s_ashr_i32 s9, s8, 31                                      // 000000001788: 86099F08
	s_movk_i32 s3, 0x12c0                                      // 00000000178C: B00312C0
	s_lshl_b64 s[6:7], s[8:9], 2                               // 000000001790: 84868208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001794: BF8704D9
	s_add_u32 s18, s1, s6                                      // 000000001798: 80120601
	s_addc_u32 s19, s10, s7                                    // 00000000179C: 8213070A
	s_ashr_i32 s15, s14, 31                                    // 0000000017A0: 860F9F0E
	s_movk_i32 s1, 0x1310                                      // 0000000017A4: B0011310
	s_lshl_b64 s[6:7], s[14:15], 2                             // 0000000017A8: 8486820E
	s_add_u32 s34, s4, s6                                      // 0000000017AC: 80220604
	s_addc_u32 s35, s5, s7                                     // 0000000017B0: 82230705
	s_load_b256 s[4:11], s[18:19], null                        // 0000000017B4: F40C0109 F8000000
	s_load_b32 s15, s[34:35], null                             // 0000000017BC: F40003D1 F8000000
	s_clause 0x3                                               // 0000000017C4: BF850003
	s_load_b128 s[44:47], s[18:19], 0x28                       // 0000000017C8: F4080B09 F8000028
	s_load_b128 s[40:43], s[18:19], 0x50                       // 0000000017D0: F4080A09 F8000050
	s_load_b128 s[36:39], s[18:19], 0x168                      // 0000000017D8: F4080909 F8000168
	s_load_b128 s[28:31], s[18:19], 0x190                      // 0000000017E0: F4080709 F8000190
	s_load_b32 s48, s[34:35], 0x320                            // 0000000017E8: F4000C11 F8000320
	s_clause 0x1                                               // 0000000017F0: BF850001
	s_load_b128 s[24:27], s[18:19], 0x1b8                      // 0000000017F4: F4080609 F80001B8
	s_load_b128 s[20:23], s[18:19], 0x2d0                      // 0000000017FC: F4080509 F80002D0
	s_clause 0x9                                               // 000000001804: BF850009
	s_load_b32 s49, s[34:35], 0x640                            // 000000001808: F4000C51 F8000640
	s_load_b32 s50, s[34:35], 0x960                            // 000000001810: F4000C91 F8000960
	s_load_b32 s51, s[34:35], 0xc80                            // 000000001818: F4000CD1 F8000C80
	s_load_b32 s57, s[34:35], 0x50                             // 000000001820: F4000E51 F8000050
	s_load_b32 s58, s[34:35], 0xa0                             // 000000001828: F4000E91 F80000A0
	s_load_b32 s59, s[34:35], 0xf0                             // 000000001830: F4000ED1 F80000F0
	s_load_b32 s60, s[34:35], 0x140                            // 000000001838: F4000F11 F8000140
	s_load_b32 s61, s[34:35], 0x190                            // 000000001840: F4000F51 F8000190
	s_load_b32 s62, s[34:35], 0x1e0                            // 000000001848: F4000F91 F80001E0
	s_load_b32 s63, s[34:35], 0x230                            // 000000001850: F4000FD1 F8000230
	s_waitcnt lgkmcnt(0)                                       // 000000001858: BF89FC07
	v_fma_f32 v0, s4, s15, 0                                   // 00000000185C: D6130000 02001E04
	s_clause 0x5                                               // 000000001864: BF850005
	s_load_b32 s4, s[34:35], s3 offset:0x320                   // 000000001868: F4000111 06000320
	s_load_b32 s15, s[34:35], s1 offset:0x320                  // 000000001870: F40003D1 02000320
	s_load_b32 s64, s[34:35], s13 offset:0x320                 // 000000001878: F4001011 1A000320
	s_load_b32 s65, s[34:35], s33 offset:0x320                 // 000000001880: F4001051 42000320
	s_load_b32 s66, s[34:35], s56 offset:0x320                 // 000000001888: F4001091 70000320
	s_load_b32 s67, s[34:35], 0x2d0                            // 000000001890: F40010D1 F80002D0
	v_fmac_f32_e64 v0, s44, s48                                // 000000001898: D52B0000 0000602C
	s_load_b32 s44, s[18:19], 0x74                             // 0000000018A0: F4000B09 F8000074
	s_clause 0x2                                               // 0000000018A8: BF850002
	s_load_b32 s68, s[34:35], 0xfa0                            // 0000000018AC: F4001111 F8000FA0
	s_load_b32 s3, s[34:35], s3 offset:0x640                   // 0000000018B4: F40000D1 06000640
	s_load_b32 s69, s[34:35], 0x5f0                            // 0000000018BC: F4001151 F80005F0
	v_fmac_f32_e64 v0, s40, s49                                // 0000000018C4: D52B0000 00006228
	s_clause 0x5                                               // 0000000018CC: BF850005
	s_load_b32 s40, s[34:35], 0x9b0                            // 0000000018D0: F4000A11 F80009B0
	s_load_b32 s70, s[34:35], 0xa00                            // 0000000018D8: F4001191 F8000A00
	s_load_b32 s71, s[34:35], 0xa50                            // 0000000018E0: F40011D1 F8000A50
	s_load_b32 s72, s[34:35], 0xaa0                            // 0000000018E8: F4001211 F8000AA0
	s_load_b32 s73, s[34:35], 0xaf0                            // 0000000018F0: F4001251 F8000AF0
	s_load_b32 s74, s[34:35], 0x910                            // 0000000018F8: F4001291 F8000910
	v_fmac_f32_e64 v0, s36, s50                                // 000000001900: D52B0000 00006424
	s_clause 0x4                                               // 000000001908: BF850004
	s_load_b32 s36, s[34:35], 0x12c0                           // 00000000190C: F4000911 F80012C0
	s_load_b32 s75, s[34:35], 0xcd0                            // 000000001914: F40012D1 F8000CD0
	s_load_b32 s76, s[34:35], 0xd20                            // 00000000191C: F4001311 F8000D20
	s_load_b32 s77, s[34:35], 0xd70                            // 000000001924: F4001351 F8000D70
	s_load_b32 s78, s[34:35], 0xc30                            // 00000000192C: F4001391 F8000C30
	v_fmac_f32_e64 v0, s28, s51                                // 000000001934: D52B0000 0000661C
	s_load_b128 s[48:51], s[18:19], 0x2f8                      // 00000000193C: F4080C09 F80002F8
	s_movk_i32 s28, 0xa0                                       // 000000001944: B01C00A0
	s_load_b128 s[52:55], s[18:19], 0x320                      // 000000001948: F4080D09 F8000320
	s_load_b32 s28, s[34:35], s28 offset:0xfa0                 // 000000001950: F4000711 38000FA0
	s_load_b32 s79, s[18:19], 0x1b4                            // 000000001958: F40013C9 F80001B4
	s_load_b32 s80, s[34:35], 0xf50                            // 000000001960: F4001411 F8000F50
	s_waitcnt lgkmcnt(0)                                       // 000000001968: BF89FC07
	v_fmac_f32_e64 v0, s24, s68                                // 00000000196C: D52B0000 00008818
	s_clause 0x6                                               // 000000001974: BF850006
	s_load_b32 s24, s[34:35], 0x1310                           // 000000001978: F4000611 F8001310
	s_load_b32 s68, s[34:35], 0x1360                           // 000000001980: F4001111 F8001360
	s_load_b32 s81, s[34:35], 0x13b0                           // 000000001988: F4001451 F80013B0
	s_load_b32 s82, s[34:35], 0x1400                           // 000000001990: F4001491 F8001400
	s_load_b32 s83, s[34:35], 0x1450                           // 000000001998: F40014D1 F8001450
	s_load_b32 s84, s[34:35], 0x14a0                           // 0000000019A0: F4001511 F80014A0
	s_load_b32 s85, s[34:35], 0x14f0                           // 0000000019A8: F4001551 F80014F0
	v_fmac_f32_e64 v0, s20, s36                                // 0000000019B0: D52B0000 00004814
	s_load_b32 s20, s[34:35], 0x370                            // 0000000019B8: F4000511 F8000370
	s_load_b32 s86, s[18:19], 0x2f4                            // 0000000019C0: F4001589 F80002F4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019C8: BF870091
	v_fmac_f32_e64 v0, s48, s4                                 // 0000000019CC: D52B0000 00000830
	v_fmac_f32_e64 v0, s52, s3                                 // 0000000019D4: D52B0000 00000634
	s_movk_i32 s3, 0x1590                                      // 0000000019DC: B0031590
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019E0: BF870001
	v_fmac_f32_e64 v0, s5, s57                                 // 0000000019E4: D52B0000 00007205
	s_clause 0x2                                               // 0000000019EC: BF850002
	s_load_b32 s4, s[34:35], 0x3c0                             // 0000000019F0: F4000111 F80003C0
	s_load_b32 s5, s[34:35], 0x410                             // 0000000019F8: F4000151 F8000410
	s_load_b32 s48, s[34:35], s3 offset:0x320                  // 000000001A00: F4000C11 06000320
	s_waitcnt lgkmcnt(0)                                       // 000000001A08: BF89FC07
	v_fmac_f32_e64 v0, s45, s20                                // 000000001A0C: D52B0000 0000282D
	s_clause 0x7                                               // 000000001A14: BF850007
	s_load_b32 s20, s[34:35], 0x6e0                            // 000000001A18: F4000511 F80006E0
	s_load_b32 s36, s[34:35], 0x730                            // 000000001A20: F4000911 F8000730
	s_load_b32 s45, s[34:35], 0x780                            // 000000001A28: F4000B51 F8000780
	s_load_b32 s52, s[34:35], 0x7d0                            // 000000001A30: F4000D11 F80007D0
	s_load_b32 s57, s[34:35], 0x820                            // 000000001A38: F4000E51 F8000820
	s_load_b32 s87, s[34:35], 0x870                            // 000000001A40: F40015D1 F8000870
	s_load_b32 s88, s[34:35], s3 offset:0x640                  // 000000001A48: F4001611 06000640
	s_load_b32 s3, s[34:35], 0x690                             // 000000001A50: F40000D1 F8000690
	s_waitcnt lgkmcnt(0)                                       // 000000001A58: BF89FC07
	v_fmac_f32_e64 v0, s41, s3                                 // 000000001A5C: D52B0000 00000629
	s_movk_i32 s3, 0x2d0                                       // 000000001A64: B00302D0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A68: BF870091
	v_fmac_f32_e64 v0, s37, s40                                // 000000001A6C: D52B0000 00005025
	v_fmac_f32_e64 v0, s29, s75                                // 000000001A74: D52B0000 0000961D
	s_clause 0x1                                               // 000000001A7C: BF850001
	s_load_b32 s75, s[34:35], s3 offset:0xfa0                  // 000000001A80: F40012D1 06000FA0
	s_load_b32 s3, s[34:35], 0xff0                             // 000000001A88: F40000D1 F8000FF0
	s_waitcnt lgkmcnt(0)                                       // 000000001A90: BF89FC07
	v_fmac_f32_e64 v0, s25, s3                                 // 000000001A94: D52B0000 00000619
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A9C: BF870091
	v_fmac_f32_e64 v0, s21, s24                                // 000000001AA0: D52B0000 00003015
	v_fmac_f32_e64 v0, s49, s15                                // 000000001AA8: D52B0000 00001E31
	s_clause 0x3                                               // 000000001AB0: BF850003
	s_load_b32 s1, s[34:35], s1 offset:0x640                   // 000000001AB4: F4000051 02000640
	s_load_b32 s3, s[34:35], s13 offset:0x640                  // 000000001ABC: F40000D1 1A000640
	s_load_b32 s13, s[34:35], s33 offset:0x640                 // 000000001AC4: F4000351 42000640
	s_load_b32 s15, s[34:35], s56 offset:0x640                 // 000000001ACC: F40003D1 70000640
	s_movk_i32 s33, 0x1450                                     // 000000001AD4: B0211450
	s_movk_i32 s49, 0x14a0                                     // 000000001AD8: B03114A0
	s_movk_i32 s56, 0x14f0                                     // 000000001ADC: B03814F0
	s_clause 0x3                                               // 000000001AE0: BF850003
	s_load_b32 s90, s[34:35], s33 offset:0x640                 // 000000001AE4: F4001691 42000640
	s_load_b32 s91, s[34:35], s49 offset:0x640                 // 000000001AEC: F40016D1 62000640
	s_load_b32 s92, s[34:35], s56 offset:0x640                 // 000000001AF4: F4001711 70000640
	s_load_b32 s93, s[34:35], s89 offset:0x640                 // 000000001AFC: F4001751 B2000640
	s_waitcnt lgkmcnt(0)                                       // 000000001B04: BF89FC07
	v_fmac_f32_e64 v0, s53, s1                                 // 000000001B08: D52B0000 00000235
	s_movk_i32 s1, 0xf0                                        // 000000001B10: B00100F0
	s_load_b32 s1, s[34:35], s1 offset:0xfa0                   // 000000001B14: F4000051 02000FA0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B1C: BF8700A1
	v_fmac_f32_e64 v0, s6, s58                                 // 000000001B20: D52B0000 00007406
	s_movk_i32 s6, 0x230                                       // 000000001B28: B0060230
	v_fmac_f32_e64 v0, s46, s4                                 // 000000001B2C: D52B0000 0000082E
	s_movk_i32 s4, 0x190                                       // 000000001B34: B0040190
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B38: BF870091
	v_fmac_f32_e64 v0, s42, s20                                // 000000001B3C: D52B0000 0000282A
	v_fmac_f32_e64 v0, s38, s70                                // 000000001B44: D52B0000 00008C26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B4C: BF870091
	v_fmac_f32_e64 v0, s30, s76                                // 000000001B50: D52B0000 0000981E
	v_fmac_f32_e64 v0, s26, s28                                // 000000001B58: D52B0000 0000381A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B60: BF870091
	v_fmac_f32_e64 v0, s22, s68                                // 000000001B64: D52B0000 00008816
	v_fmac_f32_e64 v0, s50, s64                                // 000000001B6C: D52B0000 00008032
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B74: BF8700A1
	v_fmac_f32_e64 v0, s54, s3                                 // 000000001B78: D52B0000 00000636
	s_movk_i32 s3, 0x140                                       // 000000001B80: B0030140
	v_fmac_f32_e64 v0, s7, s59                                 // 000000001B84: D52B0000 00007607
	s_movk_i32 s7, 0x280                                       // 000000001B8C: B0070280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B90: BF870001
	v_fmac_f32_e64 v0, s47, s5                                 // 000000001B94: D52B0000 00000A2F
	s_movk_i32 s5, 0x1e0                                       // 000000001B9C: B00501E0
	s_clause 0x4                                               // 000000001BA0: BF850004
	s_load_b32 s3, s[34:35], s3 offset:0xfa0                   // 000000001BA4: F40000D1 06000FA0
	s_load_b32 s46, s[34:35], s4 offset:0xfa0                  // 000000001BAC: F4000B91 08000FA0
	s_load_b32 s47, s[34:35], s5 offset:0xfa0                  // 000000001BB4: F4000BD1 0A000FA0
	s_load_b32 s50, s[34:35], s6 offset:0xfa0                  // 000000001BBC: F4000C91 0C000FA0
	s_load_b32 s53, s[34:35], s7 offset:0xfa0                  // 000000001BC4: F4000D51 0E000FA0
	s_load_b64 s[4:5], s[18:19], 0x38                          // 000000001BCC: F4040109 F8000038
	v_fmac_f32_e64 v0, s43, s36                                // 000000001BD4: D52B0000 0000482B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BDC: BF870091
	v_fmac_f32_e64 v0, s39, s71                                // 000000001BE0: D52B0000 00008E27
	v_fmac_f32_e64 v0, s31, s77                                // 000000001BE8: D52B0000 00009A1F
	s_waitcnt lgkmcnt(0)                                       // 000000001BF0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001BF4: BF8700A1
	v_fmac_f32_e64 v0, s27, s1                                 // 000000001BF8: D52B0000 0000021B
	s_load_b32 s1, s[34:35], 0x460                             // 000000001C00: F4000051 F8000460
	v_fmac_f32_e64 v0, s23, s81                                // 000000001C08: D52B0000 0000A217
	s_clause 0x3                                               // 000000001C10: BF850003
	s_load_b64 s[6:7], s[18:19], 0x60                          // 000000001C14: F4040189 F8000060
	s_load_b64 s[20:21], s[18:19], 0x1a0                       // 000000001C1C: F4040509 F80001A0
	s_load_b64 s[22:23], s[18:19], 0x184                       // 000000001C24: F4040589 F8000184
	s_load_b64 s[24:25], s[18:19], 0x178                       // 000000001C2C: F4040609 F8000178
	v_fmac_f32_e64 v0, s51, s65                                // 000000001C34: D52B0000 00008233
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001C3C: BF870001
	v_fmac_f32_e64 v0, s55, s13                                // 000000001C40: D52B0000 00001A37
	s_clause 0x4                                               // 000000001C48: BF850004
	s_load_b32 s13, s[34:35], 0x4b0                            // 000000001C4C: F4000351 F80004B0
	s_load_b32 s51, s[34:35], 0x500                            // 000000001C54: F4000CD1 F8000500
	s_load_b32 s54, s[34:35], 0x550                            // 000000001C5C: F4000D91 F8000550
	s_load_b32 s55, s[34:35], 0x5a0                            // 000000001C64: F4000DD1 F80005A0
	s_load_b32 s58, s[34:35], 0xdc0                            // 000000001C6C: F4000E91 F8000DC0
	v_fmac_f32_e64 v0, s8, s60                                 // 000000001C74: D52B0000 00007808
	s_clause 0x7                                               // 000000001C7C: BF850007
	s_load_b32 s8, s[18:19], 0x40                              // 000000001C80: F4000209 F8000040
	s_load_b64 s[26:27], s[18:19], 0x1d4                       // 000000001C88: F4040689 F80001D4
	s_load_b64 s[28:29], s[18:19], 0x1c8                       // 000000001C90: F4040709 F80001C8
	s_load_b64 s[30:31], s[18:19], 0x44                        // 000000001C98: F4040789 F8000044
	s_load_b64 s[36:37], s[18:19], 0x2e0                       // 000000001CA0: F4040909 F80002E0
	s_load_b64 s[38:39], s[18:19], 0x330                       // 000000001CA8: F4040989 F8000330
	s_load_b64 s[40:41], s[18:19], 0x314                       // 000000001CB0: F4040A09 F8000314
	s_load_b64 s[42:43], s[18:19], 0x308                       // 000000001CB8: F4040A89 F8000308
	s_waitcnt lgkmcnt(0)                                       // 000000001CC0: BF89FC07
	v_fmac_f32_e64 v0, s4, s1                                  // 000000001CC4: D52B0000 00000204
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CCC: BF870091
	v_fmac_f32_e64 v0, s6, s45                                 // 000000001CD0: D52B0000 00005A06
	v_fmac_f32_e64 v0, s24, s72                                // 000000001CD8: D52B0000 00009018
	s_clause 0x3                                               // 000000001CE0: BF850003
	s_load_b32 s1, s[34:35], 0xe10                             // 000000001CE4: F4000051 F8000E10
	s_load_b32 s24, s[34:35], 0xe60                            // 000000001CEC: F4000611 F8000E60
	s_load_b32 s45, s[34:35], 0xeb0                            // 000000001CF4: F4000B51 F8000EB0
	s_load_b32 s59, s[34:35], 0xf00                            // 000000001CFC: F4000ED1 F8000F00
	v_fmac_f32_e64 v0, s20, s58                                // 000000001D04: D52B0000 00007414
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D0C: BF8700A1
	v_fmac_f32_e64 v0, s28, s3                                 // 000000001D10: D52B0000 0000061C
	s_load_b32 s3, s[34:35], s33 offset:0x320                  // 000000001D18: F40000D1 42000320
	v_fmac_f32_e64 v0, s36, s82                                // 000000001D20: D52B0000 0000A424
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D28: BF870091
	v_fmac_f32_e64 v0, s42, s66                                // 000000001D2C: D52B0000 0000842A
	v_fmac_f32_e64 v0, s38, s15                                // 000000001D34: D52B0000 00001E26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D3C: BF870091
	v_fmac_f32_e64 v0, s9, s61                                 // 000000001D40: D52B0000 00007A09
	v_fmac_f32_e64 v0, s5, s13                                 // 000000001D48: D52B0000 00001A05
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D50: BF870091
	v_fmac_f32_e64 v0, s7, s52                                 // 000000001D54: D52B0000 00006807
	v_fmac_f32_e64 v0, s25, s73                                // 000000001D5C: D52B0000 00009219
	s_waitcnt lgkmcnt(0)                                       // 000000001D64: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001D68: BF870001
	v_fmac_f32_e64 v0, s21, s1                                 // 000000001D6C: D52B0000 00000215
	s_clause 0x2                                               // 000000001D74: BF850002
	s_load_b32 s1, s[34:35], s49 offset:0x320                  // 000000001D78: F4000051 62000320
	s_load_b32 s13, s[34:35], s56 offset:0x320                 // 000000001D80: F4000351 70000320
	s_load_b32 s15, s[34:35], s89 offset:0x320                 // 000000001D88: F40003D1 B2000320
	s_load_b32 s6, s[18:19], 0x68                              // 000000001D90: F4000189 F8000068
	v_fmac_f32_e64 v0, s29, s46                                // 000000001D98: D52B0000 00005C1D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DA0: BF870091
	v_fmac_f32_e64 v0, s37, s83                                // 000000001DA4: D52B0000 0000A625
	v_fmac_f32_e64 v0, s43, s3                                 // 000000001DAC: D52B0000 0000062B
	s_load_b32 s3, s[18:19], 0x180                             // 000000001DB4: F40000C9 F8000180
	s_load_b32 s7, s[34:35], 0xb40                             // 000000001DBC: F40001D1 F8000B40
	s_load_b32 s9, s[18:19], 0x1a8                             // 000000001DC4: F4000249 F80001A8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DCC: BF870091
	v_fmac_f32_e64 v0, s39, s90                                // 000000001DD0: D52B0000 0000B427
	v_fmac_f32_e64 v0, s10, s62                                // 000000001DD8: D52B0000 00007C0A
	s_clause 0x3                                               // 000000001DE0: BF850003
	s_load_b32 s10, s[18:19], 0x1d0                            // 000000001DE4: F4000289 F80001D0
	s_load_b32 s20, s[18:19], 0x2e8                            // 000000001DEC: F4000509 F80002E8
	s_load_b64 s[4:5], s[18:19], 0x6c                          // 000000001DF4: F4040109 F800006C
	s_load_b32 s21, s[18:19], 0x310                            // 000000001DFC: F4000549 F8000310
	s_clause 0x1                                               // 000000001E04: BF850001
	s_load_b32 s25, s[34:35], 0xb90                            // 000000001E08: F4000651 F8000B90
	s_load_b32 s28, s[34:35], 0xbe0                            // 000000001E10: F4000711 F8000BE0
	v_fmac_f32_e64 v0, s8, s51                                 // 000000001E18: D52B0000 00006608
	s_waitcnt lgkmcnt(0)                                       // 000000001E20: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E24: BF870091
	v_fmac_f32_e64 v0, s6, s57                                 // 000000001E28: D52B0000 00007206
	v_fmac_f32_e64 v0, s3, s7                                  // 000000001E30: D52B0000 00000E03
	s_clause 0x1                                               // 000000001E38: BF850001
	s_load_b32 s3, s[18:19], 0x338                             // 000000001E3C: F40000C9 F8000338
	s_load_b64 s[6:7], s[18:19], 0x1ac                         // 000000001E44: F4040189 F80001AC
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E4C: BF8700A1
	v_fmac_f32_e64 v0, s9, s24                                 // 000000001E50: D52B0000 00003009
	s_load_b64 s[8:9], s[18:19], 0x2ec                         // 000000001E58: F4040209 F80002EC
	v_fmac_f32_e64 v0, s10, s47                                // 000000001E60: D52B0000 00005E0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E68: BF870091
	v_fmac_f32_e64 v0, s20, s84                                // 000000001E6C: D52B0000 0000A814
	v_fmac_f32_e64 v0, s21, s1                                 // 000000001E74: D52B0000 00000215
	s_load_b64 s[20:21], s[18:19], 0x33c                       // 000000001E7C: F4040509 F800033C
	s_waitcnt lgkmcnt(0)                                       // 000000001E84: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E88: BF870091
	v_fmac_f32_e64 v0, s3, s91                                 // 000000001E8C: D52B0000 0000B603
	v_fmac_f32_e64 v0, s11, s63                                // 000000001E94: D52B0000 00007E0B
	s_load_b64 s[10:11], s[18:19], 0x20                        // 000000001E9C: F4040289 F8000020
	s_clause 0x1                                               // 000000001EA4: BF850001
	s_load_b32 s1, s[34:35], 0x280                             // 000000001EA8: F4000051 F8000280
	s_load_b32 s3, s[34:35], 0x8c0                             // 000000001EB0: F40000D1 F80008C0
	v_fmac_f32_e64 v0, s30, s54                                // 000000001EB8: D52B0000 00006C1E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001EC0: BF870091
	v_fmac_f32_e64 v0, s4, s87                                 // 000000001EC4: D52B0000 0000AE04
	v_fmac_f32_e64 v0, s22, s25                                // 000000001ECC: D52B0000 00003216
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001ED4: BF870091
	v_fmac_f32_e64 v0, s6, s45                                 // 000000001ED8: D52B0000 00005A06
	v_fmac_f32_e64 v0, s26, s50                                // 000000001EE0: D52B0000 0000641A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001EE8: BF870091
	v_fmac_f32_e64 v0, s8, s85                                 // 000000001EEC: D52B0000 0000AA08
	v_fmac_f32_e64 v0, s40, s13                                // 000000001EF4: D52B0000 00001A28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001EFC: BF8700A1
	v_fmac_f32_e64 v0, s20, s92                                // 000000001F00: D52B0000 0000B814
	s_waitcnt lgkmcnt(0)                                       // 000000001F08: BF89FC07
	v_fmac_f32_e64 v0, s10, s1                                 // 000000001F0C: D52B0000 0000020A
	s_load_b32 s1, s[34:35], 0x1540                            // 000000001F14: F4000051 F8001540
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001F1C: BF870091
	v_fmac_f32_e64 v0, s31, s55                                // 000000001F20: D52B0000 00006E1F
	v_fmac_f32_e64 v0, s5, s3                                  // 000000001F28: D52B0000 00000605
	s_load_b32 s3, s[18:19], 0x4c                              // 000000001F30: F40000C9 F800004C
	s_load_b32 s6, s[34:35], 0x1590                            // 000000001F38: F4000191 F8001590
	s_load_b32 s4, s[18:19], 0x1dc                             // 000000001F40: F4000109 F80001DC
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001F48: BF870091
	v_fmac_f32_e64 v0, s23, s28                                // 000000001F4C: D52B0000 00003817
	v_fmac_f32_e64 v0, s7, s59                                 // 000000001F54: D52B0000 00007607
	s_load_b32 s7, s[18:19], 0x31c                             // 000000001F5C: F40001C9 F800031C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001F64: BF8700A1
	v_fmac_f32_e64 v0, s27, s53                                // 000000001F68: D52B0000 00006A1B
	s_waitcnt lgkmcnt(0)                                       // 000000001F70: BF89FC07
	v_fmac_f32_e64 v0, s9, s1                                  // 000000001F74: D52B0000 00000209
	s_load_b32 s1, s[18:19], 0x18c                             // 000000001F7C: F4000049 F800018C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001F84: BF870091
	v_fmac_f32_e64 v0, s41, s15                                // 000000001F88: D52B0000 00001E29
	v_fmac_f32_e64 v0, s21, s93                                // 000000001F90: D52B0000 0000BA15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001F98: BF870091
	v_fmac_f32_e64 v0, s11, s67                                // 000000001F9C: D52B0000 0000860B
	v_fmac_f32_e64 v0, s3, s69                                 // 000000001FA4: D52B0000 00008A03
	s_ashr_i32 s3, s2, 31                                      // 000000001FAC: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001FB0: BF870099
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001FB4: 84828202
	v_fmac_f32_e64 v0, s44, s74                                // 000000001FB8: D52B0000 0000942C
	s_add_u32 s8, s16, s2                                      // 000000001FC0: 80080210
	s_addc_u32 s9, s17, s3                                     // 000000001FC4: 82090311
	s_waitcnt lgkmcnt(0)                                       // 000000001FC8: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001FCC: BF8700A1
	v_fmac_f32_e64 v0, s1, s78                                 // 000000001FD0: D52B0000 00009C01
	s_load_b32 s1, s[18:19], 0x344                             // 000000001FD8: F4000049 F8000344
	v_fmac_f32_e64 v0, s79, s80                                // 000000001FE0: D52B0000 0000A04F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001FE8: BF8704A1
	v_fmac_f32_e64 v0, s4, s75                                 // 000000001FEC: D52B0000 00009604
	s_mul_i32 s4, s14, 49                                      // 000000001FF4: 9604B10E
	s_ashr_i32 s5, s4, 31                                      // 000000001FF8: 86059F04
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001FFC: BF8704A1
	v_fmac_f32_e64 v0, s86, s6                                 // 000000002000: D52B0000 00000C56
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000002008: 84828204
	s_add_u32 s4, s8, s2                                       // 00000000200C: 80040208
	s_addc_u32 s5, s9, s3                                      // 000000002010: 82050309
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000002014: BF8704A1
	v_fmac_f32_e64 v0, s7, s48                                 // 000000002018: D52B0000 00006007
	s_ashr_i32 s13, s12, 31                                    // 000000002020: 860D9F0C
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000002024: 8482820C
	s_waitcnt lgkmcnt(0)                                       // 000000002028: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000202C: BF8700C1
	v_fmac_f32_e64 v0, s1, s88                                 // 000000002030: D52B0000 0000B001
	s_add_u32 s2, s4, s2                                       // 000000002038: 80020204
	s_addc_u32 s3, s5, s3                                      // 00000000203C: 82030305
	s_ashr_i32 s1, s0, 31                                      // 000000002040: 86019F00
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000002044: CA140080 01000080
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000204C: 84808200
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002050: BF870009
	s_add_u32 s0, s2, s0                                       // 000000002054: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000002058: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 00000000205C: DC6A0000 00000001
	s_nop 0                                                    // 000000002064: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000002068: BFB60003
	s_endpgm                                                   // 00000000206C: BFB00000
