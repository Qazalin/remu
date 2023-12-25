
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_7_7_7_4_3_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[10:11], s[0:1], 0x10                          // 00000000170C: F4040280 F8000010
	s_mul_hi_i32 s2, s13, 0x92492493                           // 000000001714: 9702FF0D 92492493
	s_mul_i32 s0, s14, 0x51                                    // 00000000171C: 9600FF0E 00000051
	s_add_i32 s2, s2, s13                                      // 000000001724: 81020D02
	s_ashr_i32 s1, s0, 31                                      // 000000001728: 86019F00
	s_lshr_b32 s3, s2, 31                                      // 00000000172C: 85039F02
	s_ashr_i32 s2, s2, 2                                       // 000000001730: 86028202
	s_movk_i32 s33, 0x16c8                                     // 000000001734: B02116C8
	s_add_i32 s8, s2, s3                                       // 000000001738: 81080302
	s_lshl_b64 s[2:3], s[0:1], 2                               // 00000000173C: 84828200
	s_mul_i32 s0, s8, 7                                        // 000000001740: 96008708
	s_mul_i32 s8, s8, 9                                        // 000000001744: 96088908
	s_sub_i32 s12, s13, s0                                     // 000000001748: 818C000D
	s_movk_i32 s68, 0x222c                                     // 00000000174C: B044222C
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_add_u32 s1, s6, s2                                       // 000000001754: 80010206
	s_addc_u32 s6, s7, s3                                      // 000000001758: 82060307
	s_ashr_i32 s9, s8, 31                                      // 00000000175C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001760: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001764: 84828208
	s_add_u32 s1, s1, s2                                       // 000000001768: 80010201
	s_addc_u32 s7, s6, s3                                      // 00000000176C: 82070306
	s_ashr_i32 s13, s12, 31                                    // 000000001770: 860D9F0C
	s_mul_i32 s6, s15, 0x6c                                    // 000000001774: 9606FF0F 0000006C
	s_lshl_b64 s[2:3], s[12:13], 2                             // 00000000177C: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001780: BF8704B9
	s_add_u32 s8, s1, s2                                       // 000000001784: 80080201
	s_addc_u32 s9, s7, s3                                      // 000000001788: 82090307
	s_ashr_i32 s7, s6, 31                                      // 00000000178C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001790: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001794: BF870009
	s_add_u32 s6, s10, s6                                      // 000000001798: 8006060A
	s_addc_u32 s7, s11, s7                                     // 00000000179C: 8207070B
	s_load_b512 s[16:31], s[6:7], null                         // 0000000017A0: F4100403 F8000000
	s_clause 0x5                                               // 0000000017A8: BF850005
	s_load_b64 s[12:13], s[8:9], null                          // 0000000017AC: F4040304 F8000000
	s_load_b32 s1, s[8:9], 0x8                                 // 0000000017B4: F4000044 F8000008
	s_load_b64 s[10:11], s[8:9], 0x24                          // 0000000017BC: F4040284 F8000024
	s_load_b32 s69, s[8:9], s33 offset:0x8                     // 0000000017C4: F4001144 42000008
	s_load_b32 s70, s[8:9], s68 offset:0x8                     // 0000000017CC: F4001184 88000008
	s_load_b32 s36, s[8:9], 0x2c                               // 0000000017D4: F4000904 F800002C
	s_waitcnt lgkmcnt(0)                                       // 0000000017DC: BF89FC07
	v_fma_f32 v0, s12, s16, 0                                  // 0000000017E0: D6130000 0200200C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017E8: BF870001
	v_fmac_f32_e64 v0, s13, s17                                // 0000000017EC: D52B0000 0000220D
	s_clause 0x2                                               // 0000000017F4: BF850002
	s_load_b64 s[16:17], s[8:9], 0x48                          // 0000000017F8: F4040404 F8000048
	s_load_b64 s[34:35], s[8:9], s33 offset:0x24               // 000000001800: F4040884 42000024
	s_load_b64 s[12:13], s[8:9], s68 offset:0x24               // 000000001808: F4040304 88000024
	v_fmac_f32_e64 v0, s1, s18                                 // 000000001810: D52B0000 00002401
	s_load_b32 s1, s[8:9], 0x50                                // 000000001818: F4000044 F8000050
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001820: BF870091
	v_fmac_f32_e64 v0, s10, s19                                // 000000001824: D52B0000 0000260A
	v_fmac_f32_e64 v0, s11, s20                                // 00000000182C: D52B0000 0000280B
	s_clause 0x6                                               // 000000001834: BF850006
	s_load_b64 s[10:11], s[8:9], 0x144                         // 000000001838: F4040284 F8000144
	s_load_b64 s[52:53], s[8:9], s33 offset:0x48               // 000000001840: F4040D04 42000048
	s_load_b32 s71, s[8:9], s68 offset:0x2c                    // 000000001848: F40011C4 8800002C
	s_load_b32 s18, s[8:9], 0x14c                              // 000000001850: F4000484 F800014C
	s_load_b32 s72, s[8:9], s33 offset:0x50                    // 000000001858: F4001204 42000050
	s_load_b32 s73, s[8:9], s68 offset:0x50                    // 000000001860: F4001244 88000050
	s_load_b64 s[54:55], s[8:9], s68 offset:0x48               // 000000001868: F4040D84 88000048
	v_fmac_f32_e64 v0, s36, s21                                // 000000001870: D52B0000 00002A24
	s_waitcnt lgkmcnt(0)                                       // 000000001878: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000187C: BF870091
	v_fmac_f32_e64 v0, s16, s22                                // 000000001880: D52B0000 00002C10
	v_fmac_f32_e64 v0, s17, s23                                // 000000001888: D52B0000 00002E11
	s_clause 0x2                                               // 000000001890: BF850002
	s_load_b64 s[16:17], s[8:9], 0x168                         // 000000001894: F4040404 F8000168
	s_load_b64 s[56:57], s[8:9], s33 offset:0x144              // 00000000189C: F4040E04 42000144
	s_load_b64 s[58:59], s[8:9], s68 offset:0x144              // 0000000018A4: F4040E84 88000144
	v_fmac_f32_e64 v0, s1, s24                                 // 0000000018AC: D52B0000 00003001
	s_load_b32 s1, s[8:9], 0x170                               // 0000000018B4: F4000044 F8000170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018BC: BF870091
	v_fmac_f32_e64 v0, s10, s25                                // 0000000018C0: D52B0000 0000320A
	v_fmac_f32_e64 v0, s11, s26                                // 0000000018C8: D52B0000 0000340B
	s_clause 0x3                                               // 0000000018D0: BF850003
	s_load_b64 s[10:11], s[8:9], 0x18c                         // 0000000018D4: F4040284 F800018C
	s_load_b32 s74, s[8:9], s33 offset:0x168                   // 0000000018DC: F4001284 42000168
	s_load_b32 s75, s[8:9], s68 offset:0x168                   // 0000000018E4: F40012C4 88000168
	s_load_b32 s76, s[8:9], s68 offset:0x14c                   // 0000000018EC: F4001304 8800014C
	s_load_b512 s[36:51], s[6:7], 0x40                         // 0000000018F4: F4100903 F8000040
	v_fmac_f32_e64 v0, s18, s27                                // 0000000018FC: D52B0000 00003612
	s_clause 0x1                                               // 000000001904: BF850001
	s_load_b32 s18, s[8:9], 0x194                              // 000000001908: F4000484 F8000194
	s_load_b64 s[60:61], s[8:9], s68 offset:0x16c              // 000000001910: F4040F04 8800016C
	s_waitcnt lgkmcnt(0)                                       // 000000001918: BF89FC07
	v_fmac_f32_e64 v0, s16, s28                                // 00000000191C: D52B0000 00003810
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001924: BF8700A1
	v_fmac_f32_e64 v0, s17, s29                                // 000000001928: D52B0000 00003A11
	s_load_b64 s[16:17], s[8:9], 0x288                         // 000000001930: F4040404 F8000288
	v_fmac_f32_e64 v0, s1, s30                                 // 000000001938: D52B0000 00003C01
	s_load_b32 s1, s[8:9], 0x290                               // 000000001940: F4000044 F8000290
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001948: BF870091
	v_fmac_f32_e64 v0, s10, s31                                // 00000000194C: D52B0000 00003E0A
	v_fmac_f32_e64 v0, s11, s36                                // 000000001954: D52B0000 0000480B
	s_clause 0x2                                               // 00000000195C: BF850002
	s_load_b64 s[10:11], s[8:9], 0x2ac                         // 000000001960: F4040284 F80002AC
	s_load_b64 s[62:63], s[8:9], s33 offset:0x288              // 000000001968: F4040F84 42000288
	s_load_b64 s[64:65], s[8:9], s68 offset:0x288              // 000000001970: F4041004 88000288
	v_fmac_f32_e64 v0, s18, s37                                // 000000001978: D52B0000 00004A12
	s_waitcnt lgkmcnt(0)                                       // 000000001980: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001984: BF870001
	v_fmac_f32_e64 v0, s16, s38                                // 000000001988: D52B0000 00004C10
	s_clause 0x5                                               // 000000001990: BF850005
	s_load_b32 s16, s[8:9], 0x2b4                              // 000000001994: F4000404 F80002B4
	s_load_b64 s[18:19], s[8:9], 0x2d0                         // 00000000199C: F4040484 F80002D0
	s_load_b32 s77, s[8:9], s33 offset:0x290                   // 0000000019A4: F4001344 42000290
	s_load_b32 s78, s[8:9], s33 offset:0x2ac                   // 0000000019AC: F4001384 420002AC
	s_load_b32 s79, s[8:9], s68 offset:0x2ac                   // 0000000019B4: F40013C4 880002AC
	s_load_b32 s80, s[8:9], s68 offset:0x290                   // 0000000019BC: F4001404 88000290
	v_fmac_f32_e64 v0, s17, s39                                // 0000000019C4: D52B0000 00004E11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019CC: BF8700A1
	v_fmac_f32_e64 v0, s1, s40                                 // 0000000019D0: D52B0000 00005001
	s_load_b32 s1, s[8:9], 0x2d8                               // 0000000019D8: F4000044 F80002D8
	v_fmac_f32_e64 v0, s10, s41                                // 0000000019E0: D52B0000 0000520A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019E8: BF870001
	v_fmac_f32_e64 v0, s11, s42                                // 0000000019EC: D52B0000 0000540B
	s_clause 0x3                                               // 0000000019F4: BF850003
	s_load_b64 s[10:11], s[8:9], 0xb88                         // 0000000019F8: F4040284 F8000B88
	s_load_b64 s[36:37], s[8:9], 0xb64                         // 000000001A00: F4040904 F8000B64
	s_load_b64 s[66:67], s[8:9], s68 offset:0x2b0              // 000000001A08: F4041084 880002B0
	s_load_b32 s40, s[8:9], 0xb6c                              // 000000001A10: F4000A04 F8000B6C
	s_waitcnt lgkmcnt(0)                                       // 000000001A18: BF89FC07
	v_fmac_f32_e64 v0, s16, s43                                // 000000001A1C: D52B0000 00005610
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A24: BF870091
	v_fmac_f32_e64 v0, s18, s44                                // 000000001A28: D52B0000 00005812
	v_fmac_f32_e64 v0, s19, s45                                // 000000001A30: D52B0000 00005A13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001A38: BF8700B1
	v_fmac_f32_e64 v0, s1, s46                                 // 000000001A3C: D52B0000 00005C01
	s_load_b32 s1, s[8:9], 0xb90                               // 000000001A44: F4000044 F8000B90
	s_load_b512 s[16:31], s[6:7], 0x80                         // 000000001A4C: F4100403 F8000080
	v_fmac_f32_e64 v0, s36, s47                                // 000000001A54: D52B0000 00005E24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001A5C: BF8700C1
	v_fmac_f32_e64 v0, s37, s48                                // 000000001A60: D52B0000 00006025
	s_clause 0x1                                               // 000000001A68: BF850001
	s_load_b64 s[36:37], s[8:9], 0xca8                         // 000000001A6C: F4040904 F8000CA8
	s_load_b64 s[38:39], s[8:9], 0xbac                         // 000000001A74: F4040984 F8000BAC
	v_fmac_f32_e64 v0, s40, s49                                // 000000001A7C: D52B0000 00006228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A84: BF8700A1
	v_fmac_f32_e64 v0, s10, s50                                // 000000001A88: D52B0000 0000640A
	s_load_b32 s10, s[8:9], 0xbb4                              // 000000001A90: F4000284 F8000BB4
	v_fmac_f32_e64 v0, s11, s51                                // 000000001A98: D52B0000 0000660B
	s_waitcnt lgkmcnt(0)                                       // 000000001AA0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001AA4: BF8700A1
	v_fmac_f32_e64 v0, s1, s16                                 // 000000001AA8: D52B0000 00002001
	s_load_b32 s1, s[8:9], 0xcb0                               // 000000001AB0: F4000044 F8000CB0
	v_fmac_f32_e64 v0, s38, s17                                // 000000001AB8: D52B0000 00002226
	s_load_b64 s[16:17], s[8:9], 0xcf4                         // 000000001AC0: F4040404 F8000CF4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001AC8: BF8700A1
	v_fmac_f32_e64 v0, s39, s18                                // 000000001ACC: D52B0000 00002427
	s_load_b32 s18, s[8:9], 0xccc                              // 000000001AD4: F4000484 F8000CCC
	v_fmac_f32_e64 v0, s10, s19                                // 000000001ADC: D52B0000 0000260A
	s_load_b64 s[10:11], s[8:9], 0xcd0                         // 000000001AE4: F4040284 F8000CD0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001AEC: BF8700A1
	v_fmac_f32_e64 v0, s36, s20                                // 000000001AF0: D52B0000 00002824
	s_load_b32 s20, s[8:9], 0xcf0                              // 000000001AF8: F4000504 F8000CF0
	v_fmac_f32_e64 v0, s37, s21                                // 000000001B00: D52B0000 00002A25
	s_waitcnt lgkmcnt(0)                                       // 000000001B08: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B0C: BF870091
	v_fmac_f32_e64 v0, s1, s22                                 // 000000001B10: D52B0000 00002C01
	v_fmac_f32_e64 v0, s18, s23                                // 000000001B18: D52B0000 00002E12
	s_clause 0x1                                               // 000000001B20: BF850001
	s_load_b32 s1, s[8:9], 0xe10                               // 000000001B24: F4000044 F8000E10
	s_load_b64 s[18:19], s[8:9], 0xdec                         // 000000001B2C: F4040484 F8000DEC
	s_load_b512 s[36:51], s[6:7], 0xc0                         // 000000001B34: F4100903 F80000C0
	v_fmac_f32_e64 v0, s10, s24                                // 000000001B3C: D52B0000 0000300A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B44: BF8700A1
	v_fmac_f32_e64 v0, s11, s25                                // 000000001B48: D52B0000 0000320B
	s_load_b64 s[10:11], s[8:9], 0xe14                         // 000000001B50: F4040284 F8000E14
	v_fmac_f32_e64 v0, s20, s26                                // 000000001B58: D52B0000 00003414
	s_load_b32 s20, s[8:9], 0xdf4                              // 000000001B60: F4000504 F8000DF4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B68: BF870091
	v_fmac_f32_e64 v0, s16, s27                                // 000000001B6C: D52B0000 00003610
	v_fmac_f32_e64 v0, s17, s28                                // 000000001B74: D52B0000 00003811
	s_waitcnt lgkmcnt(0)                                       // 000000001B7C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B80: BF870091
	v_fmac_f32_e64 v0, s18, s29                                // 000000001B84: D52B0000 00003A12
	v_fmac_f32_e64 v0, s19, s30                                // 000000001B8C: D52B0000 00003C13
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B94: BF870001
	v_fmac_f32_e64 v0, s20, s31                                // 000000001B98: D52B0000 00003E14
	s_clause 0x2                                               // 000000001BA0: BF850002
	s_load_b32 s20, s[8:9], 0xe34                              // 000000001BA4: F4000504 F8000E34
	s_load_b64 s[16:17], s[8:9], 0xe38                         // 000000001BAC: F4040404 F8000E38
	s_load_b64 s[18:19], s[8:9], 0x16c8                        // 000000001BB4: F4040484 F80016C8
	v_fmac_f32_e64 v0, s1, s36                                 // 000000001BBC: D52B0000 00004801
	s_load_b32 s1, s[8:9], s33 offset:0x2c                     // 000000001BC4: F4000044 4200002C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BCC: BF870091
	v_fmac_f32_e64 v0, s10, s37                                // 000000001BD0: D52B0000 00004A0A
	v_fmac_f32_e64 v0, s11, s38                                // 000000001BD8: D52B0000 00004C0B
	s_load_b64 s[10:11], s[8:9], s33 offset:0x16c              // 000000001BE0: F4040284 4200016C
	s_waitcnt lgkmcnt(0)                                       // 000000001BE8: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BEC: BF870091
	v_fmac_f32_e64 v0, s20, s39                                // 000000001BF0: D52B0000 00004E14
	v_fmac_f32_e64 v0, s16, s40                                // 000000001BF8: D52B0000 00005010
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C00: BF870091
	v_fmac_f32_e64 v0, s17, s41                                // 000000001C04: D52B0000 00005211
	v_fmac_f32_e64 v0, s18, s42                                // 000000001C0C: D52B0000 00005412
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C14: BF8700A1
	v_fmac_f32_e64 v0, s19, s43                                // 000000001C18: D52B0000 00005613
	s_load_b512 s[16:31], s[6:7], 0x100                        // 000000001C20: F4100403 F8000100
	v_fmac_f32_e64 v0, s69, s44                                // 000000001C28: D52B0000 00005845
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C30: BF870091
	v_fmac_f32_e64 v0, s34, s45                                // 000000001C34: D52B0000 00005A22
	v_fmac_f32_e64 v0, s35, s46                                // 000000001C3C: D52B0000 00005C23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C44: BF8700A1
	v_fmac_f32_e64 v0, s1, s47                                 // 000000001C48: D52B0000 00005E01
	s_load_b32 s1, s[8:9], s33 offset:0x14c                    // 000000001C50: F4000044 4200014C
	v_fmac_f32_e64 v0, s52, s48                                // 000000001C58: D52B0000 00006034
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C60: BF870091
	v_fmac_f32_e64 v0, s53, s49                                // 000000001C64: D52B0000 00006235
	v_fmac_f32_e64 v0, s72, s50                                // 000000001C6C: D52B0000 00006448
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C74: BF8700A1
	v_fmac_f32_e64 v0, s56, s51                                // 000000001C78: D52B0000 00006638
	s_waitcnt lgkmcnt(0)                                       // 000000001C80: BF89FC07
	v_fmac_f32_e64 v0, s57, s16                                // 000000001C84: D52B0000 00002039
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001C8C: BF8700C1
	v_fmac_f32_e64 v0, s1, s17                                 // 000000001C90: D52B0000 00002201
	s_clause 0x1                                               // 000000001C98: BF850001
	s_load_b32 s1, s[8:9], s33 offset:0x18c                    // 000000001C9C: F4000044 4200018C
	s_load_b64 s[16:17], s[8:9], s33 offset:0x190              // 000000001CA4: F4040404 42000190
	v_fmac_f32_e64 v0, s74, s18                                // 000000001CAC: D52B0000 0000244A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CB4: BF870091
	v_fmac_f32_e64 v0, s10, s19                                // 000000001CB8: D52B0000 0000260A
	v_fmac_f32_e64 v0, s11, s20                                // 000000001CC0: D52B0000 0000280B
	s_clause 0x1                                               // 000000001CC8: BF850001
	s_load_b64 s[10:11], s[8:9], s68 offset:0x190              // 000000001CCC: F4040284 88000190
	s_load_b32 s34, s[8:9], s68 offset:0x18c                   // 000000001CD4: F4000884 8800018C
	s_waitcnt lgkmcnt(0)                                       // 000000001CDC: BF89FC07
	v_fmac_f32_e64 v0, s1, s21                                 // 000000001CE0: D52B0000 00002A01
	s_load_b32 s1, s[8:9], s33 offset:0x2d0                    // 000000001CE8: F4000044 420002D0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CF0: BF870091
	v_fmac_f32_e64 v0, s16, s22                                // 000000001CF4: D52B0000 00002C10
	v_fmac_f32_e64 v0, s17, s23                                // 000000001CFC: D52B0000 00002E11
	s_clause 0x1                                               // 000000001D04: BF850001
	s_load_b64 s[16:17], s[8:9], s33 offset:0x2b0              // 000000001D08: F4040404 420002B0
	s_load_b64 s[18:19], s[8:9], s33 offset:0x2d4              // 000000001D10: F4040484 420002D4
	s_load_b512 s[36:51], s[6:7], 0x140                        // 000000001D18: F4100903 F8000140
	s_load_b64 s[20:21], s[8:9], 0x222c                        // 000000001D20: F4040504 F800222C
	v_fmac_f32_e64 v0, s62, s24                                // 000000001D28: D52B0000 0000303E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D30: BF870091
	v_fmac_f32_e64 v0, s63, s25                                // 000000001D34: D52B0000 0000323F
	v_fmac_f32_e64 v0, s77, s26                                // 000000001D3C: D52B0000 0000344D
	s_clause 0x1                                               // 000000001D44: BF850001
	s_load_b64 s[24:25], s[8:9], s68 offset:0x2d4              // 000000001D48: F4040604 880002D4
	s_load_b32 s26, s[8:9], s68 offset:0x2d0                   // 000000001D50: F4000684 880002D0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D58: BF8700A1
	v_fmac_f32_e64 v0, s78, s27                                // 000000001D5C: D52B0000 0000364E
	s_waitcnt lgkmcnt(0)                                       // 000000001D64: BF89FC07
	v_fmac_f32_e64 v0, s16, s28                                // 000000001D68: D52B0000 00003810
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D70: BF870091
	v_fmac_f32_e64 v0, s17, s29                                // 000000001D74: D52B0000 00003A11
	v_fmac_f32_e64 v0, s1, s30                                 // 000000001D7C: D52B0000 00003C01
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D84: BF870091
	v_fmac_f32_e64 v0, s18, s31                                // 000000001D88: D52B0000 00003E12
	v_fmac_f32_e64 v0, s19, s36                                // 000000001D90: D52B0000 00004813
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D98: BF870091
	v_fmac_f32_e64 v0, s20, s37                                // 000000001D9C: D52B0000 00004A14
	v_fmac_f32_e64 v0, s21, s38                                // 000000001DA4: D52B0000 00004C15
	s_load_b256 s[16:23], s[6:7], 0x180                        // 000000001DAC: F40C0403 F8000180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DB4: BF870091
	v_fmac_f32_e64 v0, s70, s39                                // 000000001DB8: D52B0000 00004E46
	v_fmac_f32_e64 v0, s12, s40                                // 000000001DC0: D52B0000 0000500C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DC8: BF870091
	v_fmac_f32_e64 v0, s13, s41                                // 000000001DCC: D52B0000 0000520D
	v_fmac_f32_e64 v0, s71, s42                                // 000000001DD4: D52B0000 00005447
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DDC: BF870091
	v_fmac_f32_e64 v0, s54, s43                                // 000000001DE0: D52B0000 00005636
	v_fmac_f32_e64 v0, s55, s44                                // 000000001DE8: D52B0000 00005837
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DF0: BF870091
	v_fmac_f32_e64 v0, s73, s45                                // 000000001DF4: D52B0000 00005A49
	v_fmac_f32_e64 v0, s58, s46                                // 000000001DFC: D52B0000 00005C3A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E04: BF870091
	v_fmac_f32_e64 v0, s59, s47                                // 000000001E08: D52B0000 00005E3B
	v_fmac_f32_e64 v0, s76, s48                                // 000000001E10: D52B0000 0000604C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E18: BF870091
	v_fmac_f32_e64 v0, s75, s49                                // 000000001E1C: D52B0000 0000624B
	v_fmac_f32_e64 v0, s60, s50                                // 000000001E24: D52B0000 0000643C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E2C: BF8700A1
	v_fmac_f32_e64 v0, s61, s51                                // 000000001E30: D52B0000 0000663D
	s_waitcnt lgkmcnt(0)                                       // 000000001E38: BF89FC07
	v_fmac_f32_e64 v0, s34, s16                                // 000000001E3C: D52B0000 00002022
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E44: BF870091
	v_fmac_f32_e64 v0, s10, s17                                // 000000001E48: D52B0000 0000220A
	v_fmac_f32_e64 v0, s11, s18                                // 000000001E50: D52B0000 0000240B
	s_load_b128 s[8:11], s[6:7], 0x1a0                         // 000000001E58: F4080203 F80001A0
	s_mul_i32 s6, s15, 0x157                                   // 000000001E60: 9606FF0F 00000157
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E68: BF870099
	s_ashr_i32 s7, s6, 31                                      // 000000001E6C: 86079F06
	v_fmac_f32_e64 v0, s64, s19                                // 000000001E70: D52B0000 00002640
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001E78: 84868206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E7C: BF8700A9
	s_add_u32 s1, s4, s6                                       // 000000001E80: 80010604
	s_addc_u32 s6, s5, s7                                      // 000000001E84: 82060705
	v_fmac_f32_e64 v0, s65, s20                                // 000000001E88: D52B0000 00002841
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E90: BF870091
	v_fmac_f32_e64 v0, s80, s21                                // 000000001E94: D52B0000 00002A50
	v_fmac_f32_e64 v0, s79, s22                                // 000000001E9C: D52B0000 00002C4F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001EA4: BF8700A1
	v_fmac_f32_e64 v0, s66, s23                                // 000000001EA8: D52B0000 00002E42
	s_waitcnt lgkmcnt(0)                                       // 000000001EB0: BF89FC07
	v_fmac_f32_e64 v0, s67, s8                                 // 000000001EB4: D52B0000 00001043
	s_mul_i32 s8, s14, 49                                      // 000000001EBC: 9608B10E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001EC0: BF8704A1
	v_fmac_f32_e64 v0, s26, s9                                 // 000000001EC4: D52B0000 0000121A
	s_ashr_i32 s9, s8, 31                                      // 000000001ECC: 86099F08
	s_lshl_b64 s[4:5], s[8:9], 2                               // 000000001ED0: 84848208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001ED4: BF8700C1
	v_fmac_f32_e64 v0, s24, s10                                // 000000001ED8: D52B0000 00001418
	s_add_u32 s4, s1, s4                                       // 000000001EE0: 80040401
	s_addc_u32 s5, s6, s5                                      // 000000001EE4: 82050506
	s_ashr_i32 s1, s0, 31                                      // 000000001EE8: 86019F00
	v_fmac_f32_e64 v0, s25, s11                                // 000000001EEC: D52B0000 00001619
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001EF4: 84808200
	v_mov_b32_e32 v1, 0                                        // 000000001EF8: 7E020280
	s_add_u32 s0, s4, s0                                       // 000000001EFC: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001F00: 82010105
	v_max_f32_e32 v0, 0, v0                                    // 000000001F04: 20000080
	s_add_u32 s0, s0, s2                                       // 000000001F08: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001F0C: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001F10: DC6A0000 00000001
	s_nop 0                                                    // 000000001F18: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001F1C: BFB60003
	s_endpgm                                                   // 000000001F20: BFB00000
