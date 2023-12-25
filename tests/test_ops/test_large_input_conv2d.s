
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_60_63_16_5_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[10:11], s[0:1], 0x10                          // 00000000170C: F4040280 F8000010
	s_mul_hi_i32 s2, s13, 0x82082083                           // 000000001714: 9702FF0D 82082083
	s_mov_b32 s33, 0x8000                                      // 00000000171C: BEA100FF 00008000
	s_add_i32 s2, s2, s13                                      // 000000001724: 81020D02
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001728: BF870009
	s_lshr_b32 s0, s2, 31                                      // 00000000172C: 85009F02
	s_ashr_i32 s1, s2, 5                                       // 000000001730: 86018502
	s_lshl_b32 s2, s15, 16                                     // 000000001734: 8402900F
	s_add_i32 s1, s1, s0                                       // 000000001738: 81010001
	s_ashr_i32 s3, s2, 31                                      // 00000000173C: 86039F02
	s_mul_i32 s0, s1, 63                                       // 000000001740: 9600BF01
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001744: 84828202
	s_sub_i32 s8, s13, s0                                      // 000000001748: 8188000D
	s_waitcnt lgkmcnt(0)                                       // 00000000174C: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001750: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001754: 82070307
	s_lshl_b32 s2, s1, 6                                       // 000000001758: 84028601
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000175C: BF870499
	s_ashr_i32 s3, s2, 31                                      // 000000001760: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001764: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001768: BF8704D9
	s_add_u32 s1, s6, s2                                       // 00000000176C: 80010206
	s_addc_u32 s7, s7, s3                                      // 000000001770: 82070307
	s_ashr_i32 s9, s8, 31                                      // 000000001774: 86099F08
	s_mul_i32 s6, s14, 0xa0                                    // 000000001778: 9606FF0E 000000A0
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001780: 84828208
	s_add_u32 s8, s1, s2                                       // 000000001784: 80080201
	s_addc_u32 s9, s7, s3                                      // 000000001788: 82090307
	s_ashr_i32 s7, s6, 31                                      // 00000000178C: 86079F06
	s_movk_i32 s1, 0x4000                                      // 000000001790: B0014000
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001794: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001798: BF870009
	s_add_u32 s6, s10, s6                                      // 00000000179C: 8006060A
	s_addc_u32 s7, s11, s7                                     // 0000000017A0: 8207070B
	s_load_b512 s[16:31], s[6:7], null                         // 0000000017A4: F4100403 F8000000
	s_clause 0x4                                               // 0000000017AC: BF850004
	s_load_b64 s[38:39], s[8:9], null                          // 0000000017B0: F4040984 F8000000
	s_load_b64 s[10:11], s[8:9], 0x100                         // 0000000017B8: F4040284 F8000100
	s_load_b64 s[34:35], s[8:9], s1 offset:0x100               // 0000000017C0: F4040884 02000100
	s_load_b64 s[12:13], s[8:9], s33 offset:0x100              // 0000000017C8: F4040304 42000100
	s_load_b64 s[36:37], s[8:9], 0x200                         // 0000000017D0: F4040904 F8000200
	s_waitcnt lgkmcnt(0)                                       // 0000000017D8: BF89FC07
	v_fma_f32 v0, s38, s16, 0                                  // 0000000017DC: D6130000 02002026
	s_mov_b32 s38, 0x3c000                                     // 0000000017E4: BEA600FF 0003C000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017EC: BF8700A1
	v_fmac_f32_e64 v0, s39, s17                                // 0000000017F0: D52B0000 00002227
	s_load_b64 s[16:17], s[8:9], 0x300                         // 0000000017F8: F4040404 F8000300
	v_fmac_f32_e64 v0, s10, s18                                // 000000001800: D52B0000 0000240A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001808: BF870001
	v_fmac_f32_e64 v0, s11, s19                                // 00000000180C: D52B0000 0000260B
	s_clause 0x3                                               // 000000001814: BF850003
	s_load_b64 s[18:19], s[8:9], 0x400                         // 000000001818: F4040484 F8000400
	s_load_b64 s[52:53], s[8:9], s1 offset:0x200               // 000000001820: F4040D04 02000200
	s_load_b64 s[54:55], s[8:9], s33 offset:0x200              // 000000001828: F4040D84 42000200
	s_load_b64 s[10:11], s[8:9], s38 offset:0x100              // 000000001830: F4040284 4C000100
	v_fmac_f32_e64 v0, s36, s20                                // 000000001838: D52B0000 00002824
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001840: BF870001
	v_fmac_f32_e64 v0, s37, s21                                // 000000001844: D52B0000 00002A25
	s_clause 0x3                                               // 00000000184C: BF850003
	s_load_b64 s[20:21], s[8:9], 0x4000                        // 000000001850: F4040504 F8004000
	s_load_b64 s[56:57], s[8:9], s1 offset:0x300               // 000000001858: F4040E04 02000300
	s_load_b64 s[58:59], s[8:9], s33 offset:0x300              // 000000001860: F4040E84 42000300
	s_load_b64 s[72:73], s[8:9], s38 offset:0x200              // 000000001868: F4041204 4C000200
	s_waitcnt lgkmcnt(0)                                       // 000000001870: BF89FC07
	v_fmac_f32_e64 v0, s16, s22                                // 000000001874: D52B0000 00002C10
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000187C: BF870001
	v_fmac_f32_e64 v0, s17, s23                                // 000000001880: D52B0000 00002E11
	s_clause 0x2                                               // 000000001888: BF850002
	s_load_b64 s[16:17], s[8:9], s1 offset:0x400               // 00000000188C: F4040404 02000400
	s_load_b64 s[60:61], s[8:9], s33 offset:0x400              // 000000001894: F4040F04 42000400
	s_load_b64 s[70:71], s[8:9], s38 offset:0x300              // 00000000189C: F4041184 4C000300
	s_mov_b32 s1, 0xc000                                       // 0000000018A4: BE8100FF 0000C000
	s_mov_b32 s33, 0x10000                                     // 0000000018AC: BEA100FF 00010000
	v_fmac_f32_e64 v0, s18, s24                                // 0000000018B4: D52B0000 00003012
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018BC: BF870001
	v_fmac_f32_e64 v0, s19, s25                                // 0000000018C0: D52B0000 00003213
	s_clause 0x2                                               // 0000000018C8: BF850002
	s_load_b64 s[18:19], s[8:9], 0x8000                        // 0000000018CC: F4040484 F8008000
	s_load_b64 s[62:63], s[8:9], 0xc000                        // 0000000018D4: F4040F84 F800C000
	s_load_b64 s[68:69], s[8:9], s38 offset:0x400              // 0000000018DC: F4041104 4C000400
	s_load_b512 s[36:51], s[6:7], 0x40                         // 0000000018E4: F4100903 F8000040
	v_fmac_f32_e64 v0, s20, s26                                // 0000000018EC: D52B0000 00003414
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F4: BF870091
	v_fmac_f32_e64 v0, s21, s27                                // 0000000018F8: D52B0000 00003615
	v_fmac_f32_e64 v0, s34, s28                                // 000000001900: D52B0000 00003822
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001908: BF8700A1
	v_fmac_f32_e64 v0, s35, s29                                // 00000000190C: D52B0000 00003A23
	s_load_b64 s[34:35], s[8:9], s1 offset:0x200               // 000000001914: F4040884 02000200
	v_fmac_f32_e64 v0, s52, s30                                // 00000000191C: D52B0000 00003C34
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001924: BF8700A1
	v_fmac_f32_e64 v0, s53, s31                                // 000000001928: D52B0000 00003E35
	s_waitcnt lgkmcnt(0)                                       // 000000001930: BF89FC07
	v_fmac_f32_e64 v0, s56, s36                                // 000000001934: D52B0000 00004838
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000193C: BF870091
	v_fmac_f32_e64 v0, s57, s37                                // 000000001940: D52B0000 00004A39
	v_fmac_f32_e64 v0, s16, s38                                // 000000001948: D52B0000 00004C10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001950: BF870091
	v_fmac_f32_e64 v0, s17, s39                                // 000000001954: D52B0000 00004E11
	v_fmac_f32_e64 v0, s18, s40                                // 00000000195C: D52B0000 00005012
	s_mov_b32 s40, 0x14000                                     // 000000001964: BEA800FF 00014000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 00000000196C: BF8700B1
	v_fmac_f32_e64 v0, s19, s41                                // 000000001970: D52B0000 00005213
	s_load_b512 s[16:31], s[6:7], 0x80                         // 000000001978: F4100403 F8000080
	s_mov_b32 s41, 0x18000                                     // 000000001980: BEA900FF 00018000
	v_fmac_f32_e64 v0, s12, s42                                // 000000001988: D52B0000 0000540C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001990: BF8700A1
	v_fmac_f32_e64 v0, s13, s43                                // 000000001994: D52B0000 0000560D
	s_load_b64 s[12:13], s[8:9], s1 offset:0x100               // 00000000199C: F4040304 02000100
	v_fmac_f32_e64 v0, s54, s44                                // 0000000019A4: D52B0000 00005836
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019AC: BF870001
	v_fmac_f32_e64 v0, s55, s45                                // 0000000019B0: D52B0000 00005A37
	s_clause 0x3                                               // 0000000019B8: BF850003
	s_load_b64 s[36:37], s[8:9], s1 offset:0x300               // 0000000019BC: F4040904 02000300
	s_load_b64 s[38:39], s[8:9], s33 offset:0x100              // 0000000019C4: F4040984 42000100
	s_load_b64 s[52:53], s[8:9], s40 offset:0x100              // 0000000019CC: F4040D04 50000100
	s_load_b64 s[54:55], s[8:9], s41 offset:0x100              // 0000000019D4: F4040D84 52000100
	v_fmac_f32_e64 v0, s58, s46                                // 0000000019DC: D52B0000 00005C3A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019E4: BF870091
	v_fmac_f32_e64 v0, s59, s47                                // 0000000019E8: D52B0000 00005E3B
	v_fmac_f32_e64 v0, s60, s48                                // 0000000019F0: D52B0000 0000603C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019F8: BF870091
	v_fmac_f32_e64 v0, s61, s49                                // 0000000019FC: D52B0000 0000623D
	v_fmac_f32_e64 v0, s62, s50                                // 000000001A04: D52B0000 0000643E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A0C: BF8700A1
	v_fmac_f32_e64 v0, s63, s51                                // 000000001A10: D52B0000 0000663F
	s_waitcnt lgkmcnt(0)                                       // 000000001A18: BF89FC07
	v_fmac_f32_e64 v0, s12, s16                                // 000000001A1C: D52B0000 0000200C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A24: BF870001
	v_fmac_f32_e64 v0, s13, s17                                // 000000001A28: D52B0000 0000220D
	s_clause 0x3                                               // 000000001A30: BF850003
	s_load_b64 s[12:13], s[8:9], s1 offset:0x400               // 000000001A34: F4040304 02000400
	s_load_b64 s[16:17], s[8:9], s33 offset:0x200              // 000000001A3C: F4040404 42000200
	s_load_b64 s[56:57], s[8:9], s40 offset:0x200              // 000000001A44: F4040E04 50000200
	s_load_b64 s[58:59], s[8:9], s41 offset:0x200              // 000000001A4C: F4040E84 52000200
	s_mov_b32 s1, 0x1c000                                      // 000000001A54: BE8100FF 0001C000
	v_fmac_f32_e64 v0, s34, s18                                // 000000001A5C: D52B0000 00002422
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A64: BF870001
	v_fmac_f32_e64 v0, s35, s19                                // 000000001A68: D52B0000 00002623
	s_clause 0x3                                               // 000000001A70: BF850003
	s_load_b64 s[18:19], s[8:9], 0x10000                       // 000000001A74: F4040484 F8010000
	s_load_b64 s[34:35], s[8:9], s33 offset:0x300              // 000000001A7C: F4040884 42000300
	s_load_b64 s[60:61], s[8:9], s40 offset:0x300              // 000000001A84: F4040F04 50000300
	s_load_b64 s[62:63], s[8:9], s41 offset:0x300              // 000000001A8C: F4040F84 52000300
	v_fmac_f32_e64 v0, s36, s20                                // 000000001A94: D52B0000 00002824
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A9C: BF870001
	v_fmac_f32_e64 v0, s37, s21                                // 000000001AA0: D52B0000 00002A25
	s_clause 0x2                                               // 000000001AA8: BF850002
	s_load_b64 s[20:21], s[8:9], s33 offset:0x400              // 000000001AAC: F4040504 42000400
	s_load_b64 s[64:65], s[8:9], s40 offset:0x400              // 000000001AB4: F4041004 50000400
	s_load_b64 s[66:67], s[8:9], s41 offset:0x400              // 000000001ABC: F4041084 52000400
	s_mov_b32 s33, 0x20000                                     // 000000001AC4: BEA100FF 00020000
	s_waitcnt lgkmcnt(0)                                       // 000000001ACC: BF89FC07
	v_fmac_f32_e64 v0, s12, s22                                // 000000001AD0: D52B0000 00002C0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001AD8: BF870001
	v_fmac_f32_e64 v0, s13, s23                                // 000000001ADC: D52B0000 00002E0D
	s_clause 0x2                                               // 000000001AE4: BF850002
	s_load_b64 s[12:13], s[8:9], 0x14000                       // 000000001AE8: F4040304 F8014000
	s_load_b64 s[74:75], s[8:9], 0x18000                       // 000000001AF0: F4041284 F8018000
	s_load_b64 s[76:77], s[8:9], 0x1c000                       // 000000001AF8: F4041304 F801C000
	v_fmac_f32_e64 v0, s18, s24                                // 000000001B00: D52B0000 00003012
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B08: BF870091
	v_fmac_f32_e64 v0, s19, s25                                // 000000001B0C: D52B0000 00003213
	v_fmac_f32_e64 v0, s38, s26                                // 000000001B14: D52B0000 00003426
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B1C: BF8700A1
	v_fmac_f32_e64 v0, s39, s27                                // 000000001B20: D52B0000 00003627
	s_load_b512 s[36:51], s[6:7], 0xc0                         // 000000001B28: F4100903 F80000C0
	v_fmac_f32_e64 v0, s16, s28                                // 000000001B30: D52B0000 00003810
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B38: BF870091
	v_fmac_f32_e64 v0, s17, s29                                // 000000001B3C: D52B0000 00003A11
	v_fmac_f32_e64 v0, s34, s30                                // 000000001B44: D52B0000 00003C22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B4C: BF8700A1
	v_fmac_f32_e64 v0, s35, s31                                // 000000001B50: D52B0000 00003E23
	s_waitcnt lgkmcnt(0)                                       // 000000001B58: BF89FC07
	v_fmac_f32_e64 v0, s20, s36                                // 000000001B5C: D52B0000 00004814
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B64: BF8700A1
	v_fmac_f32_e64 v0, s21, s37                                // 000000001B68: D52B0000 00004A15
	s_load_b512 s[16:31], s[6:7], 0x100                        // 000000001B70: F4100403 F8000100
	v_fmac_f32_e64 v0, s12, s38                                // 000000001B78: D52B0000 00004C0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B80: BF8700A1
	v_fmac_f32_e64 v0, s13, s39                                // 000000001B84: D52B0000 00004E0D
	s_load_b64 s[12:13], s[8:9], s1 offset:0x100               // 000000001B8C: F4040304 02000100
	v_fmac_f32_e64 v0, s52, s40                                // 000000001B94: D52B0000 00005034
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B9C: BF870091
	v_fmac_f32_e64 v0, s53, s41                                // 000000001BA0: D52B0000 00005235
	v_fmac_f32_e64 v0, s56, s42                                // 000000001BA8: D52B0000 00005438
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BB0: BF870091
	v_fmac_f32_e64 v0, s57, s43                                // 000000001BB4: D52B0000 00005639
	v_fmac_f32_e64 v0, s60, s44                                // 000000001BBC: D52B0000 0000583C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC4: BF870091
	v_fmac_f32_e64 v0, s61, s45                                // 000000001BC8: D52B0000 00005A3D
	v_fmac_f32_e64 v0, s64, s46                                // 000000001BD0: D52B0000 00005C40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BD8: BF870091
	v_fmac_f32_e64 v0, s65, s47                                // 000000001BDC: D52B0000 00005E41
	v_fmac_f32_e64 v0, s74, s48                                // 000000001BE4: D52B0000 0000604A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BEC: BF870091
	v_fmac_f32_e64 v0, s75, s49                                // 000000001BF0: D52B0000 0000624B
	v_fmac_f32_e64 v0, s54, s50                                // 000000001BF8: D52B0000 00006436
	s_mov_b32 s54, 0x24000                                     // 000000001C00: BEB600FF 00024000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001C08: BF8700B1
	v_fmac_f32_e64 v0, s55, s51                                // 000000001C0C: D52B0000 00006637
	s_mov_b32 s55, 0x28000                                     // 000000001C14: BEB700FF 00028000
	s_waitcnt lgkmcnt(0)                                       // 000000001C1C: BF89FC07
	v_fmac_f32_e64 v0, s58, s16                                // 000000001C20: D52B0000 0000203A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C28: BF8700A1
	v_fmac_f32_e64 v0, s59, s17                                // 000000001C2C: D52B0000 0000223B
	s_load_b64 s[16:17], s[8:9], s1 offset:0x200               // 000000001C34: F4040404 02000200
	v_fmac_f32_e64 v0, s62, s18                                // 000000001C3C: D52B0000 0000243E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C44: BF870091
	v_fmac_f32_e64 v0, s63, s19                                // 000000001C48: D52B0000 0000263F
	v_fmac_f32_e64 v0, s66, s20                                // 000000001C50: D52B0000 00002842
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C58: BF870091
	v_fmac_f32_e64 v0, s67, s21                                // 000000001C5C: D52B0000 00002A43
	v_fmac_f32_e64 v0, s76, s22                                // 000000001C64: D52B0000 00002C4C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001C6C: BF870001
	v_fmac_f32_e64 v0, s77, s23                                // 000000001C70: D52B0000 00002E4D
	s_clause 0x3                                               // 000000001C78: BF850003
	s_load_b64 s[18:19], s[8:9], s1 offset:0x300               // 000000001C7C: F4040484 02000300
	s_load_b64 s[20:21], s[8:9], s33 offset:0x100              // 000000001C84: F4040504 42000100
	s_load_b64 s[22:23], s[8:9], s54 offset:0x100              // 000000001C8C: F4040584 6C000100
	s_load_b64 s[34:35], s[8:9], s55 offset:0x100              // 000000001C94: F4040884 6E000100
	v_fmac_f32_e64 v0, s12, s24                                // 000000001C9C: D52B0000 0000300C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001CA4: BF870001
	v_fmac_f32_e64 v0, s13, s25                                // 000000001CA8: D52B0000 0000320D
	s_clause 0x4                                               // 000000001CB0: BF850004
	s_load_b64 s[12:13], s[8:9], s1 offset:0x400               // 000000001CB4: F4040304 02000400
	s_load_b64 s[24:25], s[8:9], s33 offset:0x200              // 000000001CBC: F4040604 42000200
	s_load_b64 s[74:75], s[8:9], s54 offset:0x200              // 000000001CC4: F4041284 6C000200
	s_load_b64 s[76:77], s[8:9], s55 offset:0x200              // 000000001CCC: F4041304 6E000200
	s_load_b64 s[52:53], s[8:9], 0x20000                       // 000000001CD4: F4040D04 F8020000
	s_mov_b32 s1, 0x2c000                                      // 000000001CDC: BE8100FF 0002C000
	s_waitcnt lgkmcnt(0)                                       // 000000001CE4: BF89FC07
	v_fmac_f32_e64 v0, s16, s26                                // 000000001CE8: D52B0000 00003410
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001CF0: BF870001
	v_fmac_f32_e64 v0, s17, s27                                // 000000001CF4: D52B0000 00003611
	s_clause 0x2                                               // 000000001CFC: BF850002
	s_load_b64 s[16:17], s[8:9], s33 offset:0x300              // 000000001D00: F4040404 42000300
	s_load_b64 s[26:27], s[8:9], s54 offset:0x300              // 000000001D08: F4040684 6C000300
	s_load_b64 s[78:79], s[8:9], s55 offset:0x300              // 000000001D10: F4041384 6E000300
	s_load_b512 s[36:51], s[6:7], 0x140                        // 000000001D18: F4100903 F8000140
	v_fmac_f32_e64 v0, s18, s28                                // 000000001D20: D52B0000 00003812
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001D28: BF870001
	v_fmac_f32_e64 v0, s19, s29                                // 000000001D2C: D52B0000 00003A13
	s_clause 0x2                                               // 000000001D34: BF850002
	s_load_b64 s[18:19], s[8:9], s33 offset:0x400              // 000000001D38: F4040484 42000400
	s_load_b64 s[28:29], s[8:9], s54 offset:0x400              // 000000001D40: F4040704 6C000400
	s_load_b64 s[80:81], s[8:9], s55 offset:0x400              // 000000001D48: F4041404 6E000400
	s_mov_b32 s33, 0x30000                                     // 000000001D50: BEA100FF 00030000
	v_fmac_f32_e64 v0, s12, s30                                // 000000001D58: D52B0000 00003C0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001D60: BF870001
	v_fmac_f32_e64 v0, s13, s31                                // 000000001D64: D52B0000 00003E0D
	s_clause 0x2                                               // 000000001D6C: BF850002
	s_load_b64 s[12:13], s[8:9], 0x24000                       // 000000001D70: F4040304 F8024000
	s_load_b64 s[30:31], s[8:9], 0x28000                       // 000000001D78: F4040784 F8028000
	s_load_b64 s[82:83], s[8:9], 0x2c000                       // 000000001D80: F4041484 F802C000
	s_waitcnt lgkmcnt(0)                                       // 000000001D88: BF89FC07
	v_fmac_f32_e64 v0, s52, s36                                // 000000001D8C: D52B0000 00004834
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D94: BF8700A1
	v_fmac_f32_e64 v0, s53, s37                                // 000000001D98: D52B0000 00004A35
	s_load_b512 s[52:67], s[6:7], 0x180                        // 000000001DA0: F4100D03 F8000180
	v_fmac_f32_e64 v0, s20, s38                                // 000000001DA8: D52B0000 00004C14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DB0: BF870091
	v_fmac_f32_e64 v0, s21, s39                                // 000000001DB4: D52B0000 00004E15
	v_fmac_f32_e64 v0, s24, s40                                // 000000001DBC: D52B0000 00005018
	s_mov_b32 s40, 0x34000                                     // 000000001DC4: BEA800FF 00034000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001DCC: BF8700A1
	v_fmac_f32_e64 v0, s25, s41                                // 000000001DD0: D52B0000 00005219
	s_mov_b32 s41, 0x38000                                     // 000000001DD8: BEA900FF 00038000
	v_fmac_f32_e64 v0, s16, s42                                // 000000001DE0: D52B0000 00005410
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DE8: BF870091
	v_fmac_f32_e64 v0, s17, s43                                // 000000001DEC: D52B0000 00005611
	v_fmac_f32_e64 v0, s18, s44                                // 000000001DF4: D52B0000 00005812
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DFC: BF870091
	v_fmac_f32_e64 v0, s19, s45                                // 000000001E00: D52B0000 00005A13
	v_fmac_f32_e64 v0, s12, s46                                // 000000001E08: D52B0000 00005C0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E10: BF8700A1
	v_fmac_f32_e64 v0, s13, s47                                // 000000001E14: D52B0000 00005E0D
	s_load_b64 s[12:13], s[8:9], s1 offset:0x100               // 000000001E1C: F4040304 02000100
	v_fmac_f32_e64 v0, s22, s48                                // 000000001E24: D52B0000 00006016
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E2C: BF870091
	v_fmac_f32_e64 v0, s23, s49                                // 000000001E30: D52B0000 00006217
	v_fmac_f32_e64 v0, s74, s50                                // 000000001E38: D52B0000 0000644A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E40: BF8700A1
	v_fmac_f32_e64 v0, s75, s51                                // 000000001E44: D52B0000 0000664B
	s_waitcnt lgkmcnt(0)                                       // 000000001E4C: BF89FC07
	v_fmac_f32_e64 v0, s26, s52                                // 000000001E50: D52B0000 0000681A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E58: BF870091
	v_fmac_f32_e64 v0, s27, s53                                // 000000001E5C: D52B0000 00006A1B
	v_fmac_f32_e64 v0, s28, s54                                // 000000001E64: D52B0000 00006C1C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E6C: BF870091
	v_fmac_f32_e64 v0, s29, s55                                // 000000001E70: D52B0000 00006E1D
	v_fmac_f32_e64 v0, s30, s56                                // 000000001E78: D52B0000 0000701E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E80: BF8700A1
	v_fmac_f32_e64 v0, s31, s57                                // 000000001E84: D52B0000 0000721F
	s_load_b512 s[16:31], s[6:7], 0x1c0                        // 000000001E8C: F4100403 F80001C0
	v_fmac_f32_e64 v0, s34, s58                                // 000000001E94: D52B0000 00007422
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001E9C: BF870001
	v_fmac_f32_e64 v0, s35, s59                                // 000000001EA0: D52B0000 00007623
	s_clause 0x4                                               // 000000001EA8: BF850004
	s_load_b64 s[34:35], s[8:9], s1 offset:0x200               // 000000001EAC: F4040884 02000200
	s_load_b64 s[36:37], s[8:9], s1 offset:0x300               // 000000001EB4: F4040904 02000300
	s_load_b64 s[38:39], s[8:9], s33 offset:0x100              // 000000001EBC: F4040984 42000100
	s_load_b64 s[52:53], s[8:9], s40 offset:0x100              // 000000001EC4: F4040D04 50000100
	s_load_b64 s[54:55], s[8:9], s41 offset:0x100              // 000000001ECC: F4040D84 52000100
	v_fmac_f32_e64 v0, s76, s60                                // 000000001ED4: D52B0000 0000784C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001EDC: BF870091
	v_fmac_f32_e64 v0, s77, s61                                // 000000001EE0: D52B0000 00007A4D
	v_fmac_f32_e64 v0, s78, s62                                // 000000001EE8: D52B0000 00007C4E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001EF0: BF870091
	v_fmac_f32_e64 v0, s79, s63                                // 000000001EF4: D52B0000 00007E4F
	v_fmac_f32_e64 v0, s80, s64                                // 000000001EFC: D52B0000 00008050
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001F04: BF870091
	v_fmac_f32_e64 v0, s81, s65                                // 000000001F08: D52B0000 00008251
	v_fmac_f32_e64 v0, s82, s66                                // 000000001F10: D52B0000 00008452
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001F18: BF8700A1
	v_fmac_f32_e64 v0, s83, s67                                // 000000001F1C: D52B0000 00008653
	s_waitcnt lgkmcnt(0)                                       // 000000001F24: BF89FC07
	v_fmac_f32_e64 v0, s12, s16                                // 000000001F28: D52B0000 0000200C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001F30: BF870001
	v_fmac_f32_e64 v0, s13, s17                                // 000000001F34: D52B0000 0000220D
	s_clause 0x3                                               // 000000001F3C: BF850003
	s_load_b64 s[12:13], s[8:9], s1 offset:0x400               // 000000001F40: F4040304 02000400
	s_load_b64 s[16:17], s[8:9], s33 offset:0x200              // 000000001F48: F4040404 42000200
	s_load_b64 s[56:57], s[8:9], s40 offset:0x200              // 000000001F50: F4040E04 50000200
	s_load_b64 s[58:59], s[8:9], s41 offset:0x200              // 000000001F58: F4040E84 52000200
	v_fmac_f32_e64 v0, s34, s18                                // 000000001F60: D52B0000 00002422
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001F68: BF870001
	v_fmac_f32_e64 v0, s35, s19                                // 000000001F6C: D52B0000 00002623
	s_clause 0x3                                               // 000000001F74: BF850003
	s_load_b64 s[18:19], s[8:9], 0x30000                       // 000000001F78: F4040484 F8030000
	s_load_b64 s[34:35], s[8:9], s33 offset:0x300              // 000000001F80: F4040884 42000300
	s_load_b64 s[60:61], s[8:9], s40 offset:0x300              // 000000001F88: F4040F04 50000300
	s_load_b64 s[62:63], s[8:9], s41 offset:0x300              // 000000001F90: F4040F84 52000300
	v_fmac_f32_e64 v0, s36, s20                                // 000000001F98: D52B0000 00002824
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001FA0: BF870001
	v_fmac_f32_e64 v0, s37, s21                                // 000000001FA4: D52B0000 00002A25
	s_clause 0x2                                               // 000000001FAC: BF850002
	s_load_b64 s[20:21], s[8:9], s33 offset:0x400              // 000000001FB0: F4040504 42000400
	s_load_b64 s[64:65], s[8:9], s40 offset:0x400              // 000000001FB8: F4041004 50000400
	s_load_b64 s[66:67], s[8:9], s41 offset:0x400              // 000000001FC0: F4041084 52000400
	s_waitcnt lgkmcnt(0)                                       // 000000001FC8: BF89FC07
	v_fmac_f32_e64 v0, s12, s22                                // 000000001FCC: D52B0000 00002C0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001FD4: BF870001
	v_fmac_f32_e64 v0, s13, s23                                // 000000001FD8: D52B0000 00002E0D
	s_clause 0x2                                               // 000000001FE0: BF850002
	s_load_b64 s[12:13], s[8:9], 0x34000                       // 000000001FE4: F4040304 F8034000
	s_load_b64 s[74:75], s[8:9], 0x38000                       // 000000001FEC: F4041284 F8038000
	s_load_b64 s[8:9], s[8:9], 0x3c000                         // 000000001FF4: F4040204 F803C000
	v_fmac_f32_e64 v0, s18, s24                                // 000000001FFC: D52B0000 00003012
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002004: BF870091
	v_fmac_f32_e64 v0, s19, s25                                // 000000002008: D52B0000 00003213
	v_fmac_f32_e64 v0, s38, s26                                // 000000002010: D52B0000 00003426
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002018: BF8700A1
	v_fmac_f32_e64 v0, s39, s27                                // 00000000201C: D52B0000 00003627
	s_load_b512 s[36:51], s[6:7], 0x200                        // 000000002024: F4100903 F8000200
	v_fmac_f32_e64 v0, s16, s28                                // 00000000202C: D52B0000 00003810
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002034: BF870091
	v_fmac_f32_e64 v0, s17, s29                                // 000000002038: D52B0000 00003A11
	v_fmac_f32_e64 v0, s34, s30                                // 000000002040: D52B0000 00003C22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002048: BF8700A1
	v_fmac_f32_e64 v0, s35, s31                                // 00000000204C: D52B0000 00003E23
	s_waitcnt lgkmcnt(0)                                       // 000000002054: BF89FC07
	v_fmac_f32_e64 v0, s20, s36                                // 000000002058: D52B0000 00004814
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000002060: BF8704B1
	v_fmac_f32_e64 v0, s21, s37                                // 000000002064: D52B0000 00004A15
	s_load_b512 s[16:31], s[6:7], 0x240                        // 00000000206C: F4100403 F8000240
	s_mul_i32 s6, s15, 0x5898                                  // 000000002074: 9606FF0F 00005898
	s_ashr_i32 s7, s6, 31                                      // 00000000207C: 86079F06
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000002080: BF8704A1
	v_fmac_f32_e64 v0, s12, s38                                // 000000002084: D52B0000 00004C0C
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000208C: 84868206
	s_add_u32 s1, s4, s6                                       // 000000002090: 80010604
	s_addc_u32 s6, s5, s7                                      // 000000002094: 82060705
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002098: BF870091
	v_fmac_f32_e64 v0, s13, s39                                // 00000000209C: D52B0000 00004E0D
	v_fmac_f32_e64 v0, s52, s40                                // 0000000020A4: D52B0000 00005034
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020AC: BF870091
	v_fmac_f32_e64 v0, s53, s41                                // 0000000020B0: D52B0000 00005235
	v_fmac_f32_e64 v0, s56, s42                                // 0000000020B8: D52B0000 00005438
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020C0: BF870091
	v_fmac_f32_e64 v0, s57, s43                                // 0000000020C4: D52B0000 00005639
	v_fmac_f32_e64 v0, s60, s44                                // 0000000020CC: D52B0000 0000583C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020D4: BF870091
	v_fmac_f32_e64 v0, s61, s45                                // 0000000020D8: D52B0000 00005A3D
	v_fmac_f32_e64 v0, s64, s46                                // 0000000020E0: D52B0000 00005C40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020E8: BF870091
	v_fmac_f32_e64 v0, s65, s47                                // 0000000020EC: D52B0000 00005E41
	v_fmac_f32_e64 v0, s74, s48                                // 0000000020F4: D52B0000 0000604A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020FC: BF870091
	v_fmac_f32_e64 v0, s75, s49                                // 000000002100: D52B0000 0000624B
	v_fmac_f32_e64 v0, s54, s50                                // 000000002108: D52B0000 00006436
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002110: BF8700A1
	v_fmac_f32_e64 v0, s55, s51                                // 000000002114: D52B0000 00006637
	s_waitcnt lgkmcnt(0)                                       // 00000000211C: BF89FC07
	v_fmac_f32_e64 v0, s58, s16                                // 000000002120: D52B0000 0000203A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002128: BF870091
	v_fmac_f32_e64 v0, s59, s17                                // 00000000212C: D52B0000 0000223B
	v_fmac_f32_e64 v0, s62, s18                                // 000000002134: D52B0000 0000243E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000213C: BF870091
	v_fmac_f32_e64 v0, s63, s19                                // 000000002140: D52B0000 0000263F
	v_fmac_f32_e64 v0, s66, s20                                // 000000002148: D52B0000 00002842
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002150: BF870091
	v_fmac_f32_e64 v0, s67, s21                                // 000000002154: D52B0000 00002A43
	v_fmac_f32_e64 v0, s8, s22                                 // 00000000215C: D52B0000 00002C08
	s_mul_i32 s8, s14, 0xec4                                   // 000000002164: 9608FF0E 00000EC4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000216C: BF8704A1
	v_fmac_f32_e64 v0, s9, s23                                 // 000000002170: D52B0000 00002E09
	s_ashr_i32 s9, s8, 31                                      // 000000002178: 86099F08
	s_lshl_b64 s[4:5], s[8:9], 2                               // 00000000217C: 84848208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000002180: BF8700C1
	v_fmac_f32_e64 v0, s10, s24                                // 000000002184: D52B0000 0000300A
	s_add_u32 s4, s1, s4                                       // 00000000218C: 80040401
	s_addc_u32 s5, s6, s5                                      // 000000002190: 82050506
	s_ashr_i32 s1, s0, 31                                      // 000000002194: 86019F00
	v_fmac_f32_e64 v0, s11, s25                                // 000000002198: D52B0000 0000320B
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000021A0: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000021A4: BF8700A9
	s_add_u32 s0, s4, s0                                       // 0000000021A8: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000021AC: 82010105
	v_fmac_f32_e64 v0, s72, s26                                // 0000000021B0: D52B0000 00003448
	s_add_u32 s0, s0, s2                                       // 0000000021B8: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000021BC: 82010301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000021C0: BF870091
	v_fmac_f32_e64 v0, s73, s27                                // 0000000021C4: D52B0000 00003649
	v_fmac_f32_e64 v0, s70, s28                                // 0000000021CC: D52B0000 00003846
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000021D4: BF870091
	v_fmac_f32_e64 v0, s71, s29                                // 0000000021D8: D52B0000 00003A47
	v_fmac_f32_e64 v0, s68, s30                                // 0000000021E0: D52B0000 00003C44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000021E8: BF870091
	v_fmac_f32_e64 v0, s69, s31                                // 0000000021EC: D52B0000 00003E45
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 0000000021F4: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 0000000021FC: DC6A0000 00000001
	s_nop 0                                                    // 000000002204: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000002208: BFB60003
	s_endpgm                                                   // 00000000220C: BFB00000
