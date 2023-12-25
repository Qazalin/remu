
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_16_16_16_16_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_i32 s0, s14, 18                                      // 000000001714: 9600920E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s1, s0, 31                                      // 00000000171C: 86019F00
	s_movk_i32 s33, 0x1440                                     // 000000001720: B0211440
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001724: 84808200
	s_movk_i32 s34, 0x2370                                     // 000000001728: B0222370
	s_movk_i32 s58, 0x4bf0                                     // 00000000172C: B03A4BF0
	s_movk_i32 s56, 0x1950                                     // 000000001730: B0381950
	s_movk_i32 s57, 0x1e60                                     // 000000001734: B0391E60
	s_movk_i32 s60, 0x2880                                     // 000000001738: B03C2880
	s_movk_i32 s62, 0x2d90                                     // 00000000173C: B03E2D90
	s_movk_i32 s63, 0x32a0                                     // 000000001740: B03F32A0
	s_movk_i32 s78, 0x41d0                                     // 000000001744: B04E41D0
	s_waitcnt lgkmcnt(0)                                       // 000000001748: BF89FC07
	s_add_u32 s10, s6, s0                                      // 00000000174C: 800A0006
	s_addc_u32 s7, s7, s1                                      // 000000001750: 82070107
	s_ashr_i32 s3, s13, 31                                     // 000000001754: 86039F0D
	s_mul_i32 s6, s15, 0x90                                    // 000000001758: 9606FF0F 00000090
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001760: 84808202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001764: BF8704B9
	s_add_u32 s2, s10, s0                                      // 000000001768: 8002000A
	s_addc_u32 s3, s7, s1                                      // 00000000176C: 82030107
	s_ashr_i32 s7, s6, 31                                      // 000000001770: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001774: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001778: BF870009
	s_add_u32 s6, s8, s6                                       // 00000000177C: 80060608
	s_addc_u32 s7, s9, s7                                      // 000000001780: 82070709
	s_load_b64 s[12:13], s[2:3], null                          // 000000001784: F4040301 F8000000
	s_load_b512 s[16:31], s[6:7], null                         // 00000000178C: F4100403 F8000000
	s_clause 0x3                                               // 000000001794: BF850003
	s_load_b32 s36, s[2:3], 0x8                                // 000000001798: F4000901 F8000008
	s_load_b64 s[8:9], s[2:3], 0x48                            // 0000000017A0: F4040201 F8000048
	s_load_b32 s35, s[2:3], s33 offset:0x8                     // 0000000017A8: F40008C1 42000008
	s_load_b64 s[10:11], s[2:3], s34 offset:0x4                // 0000000017B0: F4040281 44000004
	s_waitcnt lgkmcnt(0)                                       // 0000000017B8: BF89FC07
	v_fma_f32 v0, s12, s16, 0                                  // 0000000017BC: D6130000 0200200C
	s_load_b32 s16, s[2:3], 0x50                               // 0000000017C4: F4000401 F8000050
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017CC: BF870001
	v_fmac_f32_e64 v0, s13, s17                                // 0000000017D0: D52B0000 0000220D
	s_clause 0x2                                               // 0000000017D8: BF850002
	s_load_b64 s[12:13], s[2:3], 0x90                          // 0000000017DC: F4040301 F8000090
	s_load_b64 s[52:53], s[2:3], s33 offset:0x48               // 0000000017E4: F4040D01 42000048
	s_load_b32 s59, s[2:3], s58 offset:0x8                     // 0000000017EC: F4000EC1 74000008
	v_fmac_f32_e64 v0, s36, s18                                // 0000000017F4: D52B0000 00002424
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017FC: BF870091
	v_fmac_f32_e64 v0, s8, s19                                 // 000000001800: D52B0000 00002608
	v_fmac_f32_e64 v0, s9, s20                                 // 000000001808: D52B0000 00002809
	s_clause 0x1                                               // 000000001810: BF850001
	s_load_b32 s36, s[2:3], 0x98                               // 000000001814: F4000901 F8000098
	s_load_b64 s[8:9], s[2:3], s58 offset:0x48                 // 00000000181C: F4040201 74000048
	s_waitcnt lgkmcnt(0)                                       // 000000001824: BF89FC07
	v_fmac_f32_e64 v0, s16, s21                                // 000000001828: D52B0000 00002A10
	s_clause 0xc                                               // 000000001830: BF85000C
	s_load_b32 s37, s[2:3], 0x518                              // 000000001834: F4000941 F8000518
	s_load_b64 s[16:17], s[2:3], 0x510                         // 00000000183C: F4040401 F8000510
	s_load_b32 s61, s[2:3], s58 offset:0x50                    // 000000001844: F4000F41 74000050
	s_load_b64 s[18:19], s[2:3], 0x5a0                         // 00000000184C: F4040481 F80005A0
	s_load_b64 s[20:21], s[2:3], 0x558                         // 000000001854: F4040501 F8000558
	s_load_b32 s64, s[2:3], s33 offset:0x98                    // 00000000185C: F4001001 42000098
	s_load_b32 s65, s[2:3], s56 offset:0x98                    // 000000001864: F4001041 70000098
	s_load_b32 s66, s[2:3], s57 offset:0x98                    // 00000000186C: F4001081 72000098
	s_load_b32 s67, s[2:3], s34 offset:0x98                    // 000000001874: F40010C1 44000098
	s_load_b32 s68, s[2:3], s60 offset:0x98                    // 00000000187C: F4001101 78000098
	s_load_b32 s69, s[2:3], s62 offset:0x98                    // 000000001884: F4001141 7C000098
	s_load_b32 s70, s[2:3], s63 offset:0x98                    // 00000000188C: F4001181 7E000098
	s_load_b32 s71, s[2:3], s58 offset:0x98                    // 000000001894: F40011C1 74000098
	v_fmac_f32_e64 v0, s12, s22                                // 00000000189C: D52B0000 00002C0C
	s_load_b32 s22, s[2:3], 0x560                              // 0000000018A4: F4000581 F8000560
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018AC: BF870091
	v_fmac_f32_e64 v0, s13, s23                                // 0000000018B0: D52B0000 00002E0D
	v_fmac_f32_e64 v0, s36, s24                                // 0000000018B8: D52B0000 00003024
	s_waitcnt lgkmcnt(0)                                       // 0000000018C0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018C4: BF870091
	v_fmac_f32_e64 v0, s16, s25                                // 0000000018C8: D52B0000 00003210
	v_fmac_f32_e64 v0, s17, s26                                // 0000000018D0: D52B0000 00003411
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018D8: BF8700A1
	v_fmac_f32_e64 v0, s37, s27                                // 0000000018DC: D52B0000 00003625
	s_load_b512 s[36:51], s[6:7], 0x40                         // 0000000018E4: F4100903 F8000040
	v_fmac_f32_e64 v0, s20, s28                                // 0000000018EC: D52B0000 00003814
	s_clause 0x2                                               // 0000000018F4: BF850002
	s_load_b32 s20, s[2:3], 0x5a8                              // 0000000018F8: F4000501 F80005A8
	s_load_b64 s[12:13], s[2:3], 0xa68                         // 000000001900: F4040301 F8000A68
	s_load_b64 s[16:17], s[2:3], 0xa20                         // 000000001908: F4040401 F8000A20
	v_fmac_f32_e64 v0, s21, s29                                // 000000001910: D52B0000 00003A15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001918: BF870091
	v_fmac_f32_e64 v0, s22, s30                                // 00000000191C: D52B0000 00003C16
	v_fmac_f32_e64 v0, s18, s31                                // 000000001924: D52B0000 00003E12
	s_load_b32 s18, s[2:3], 0xa28                              // 00000000192C: F4000481 F8000A28
	s_waitcnt lgkmcnt(0)                                       // 000000001934: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001938: BF8700A1
	v_fmac_f32_e64 v0, s19, s36                                // 00000000193C: D52B0000 00004813
	s_load_b32 s19, s[2:3], 0xa70                              // 000000001944: F40004C1 F8000A70
	v_fmac_f32_e64 v0, s20, s37                                // 00000000194C: D52B0000 00004A14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001954: BF870091
	v_fmac_f32_e64 v0, s16, s38                                // 000000001958: D52B0000 00004C10
	v_fmac_f32_e64 v0, s17, s39                                // 000000001960: D52B0000 00004E11
	s_clause 0x1                                               // 000000001968: BF850001
	s_load_b64 s[36:37], s[2:3], 0xf30                         // 00000000196C: F4040901 F8000F30
	s_load_b64 s[16:17], s[2:3], 0xab0                         // 000000001974: F4040401 F8000AB0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000197C: BF8700C1
	v_fmac_f32_e64 v0, s18, s40                                // 000000001980: D52B0000 00005012
	s_clause 0x1                                               // 000000001988: BF850001
	s_load_b32 s18, s[2:3], 0xab8                              // 00000000198C: F4000481 F8000AB8
	s_load_b32 s40, s[2:3], 0xf38                              // 000000001994: F4000A01 F8000F38
	v_fmac_f32_e64 v0, s12, s41                                // 00000000199C: D52B0000 0000520C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019A4: BF870001
	v_fmac_f32_e64 v0, s13, s42                                // 0000000019A8: D52B0000 0000540D
	s_clause 0x1                                               // 0000000019B0: BF850001
	s_load_b64 s[12:13], s[2:3], 0xfc0                         // 0000000019B4: F4040301 F8000FC0
	s_load_b64 s[38:39], s[2:3], 0xf78                         // 0000000019BC: F4040981 F8000F78
	s_waitcnt lgkmcnt(0)                                       // 0000000019C4: BF89FC07
	v_fmac_f32_e64 v0, s19, s43                                // 0000000019C8: D52B0000 00005613
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019D0: BF870091
	v_fmac_f32_e64 v0, s16, s44                                // 0000000019D4: D52B0000 00005810
	v_fmac_f32_e64 v0, s17, s45                                // 0000000019DC: D52B0000 00005A11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019E4: BF870091
	v_fmac_f32_e64 v0, s18, s46                                // 0000000019E8: D52B0000 00005C12
	v_fmac_f32_e64 v0, s36, s47                                // 0000000019F0: D52B0000 00005E24
	s_load_b32 s36, s[2:3], 0xf80                              // 0000000019F8: F4000901 F8000F80
	s_load_b512 s[16:31], s[6:7], 0x80                         // 000000001A00: F4100403 F8000080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A08: BF870091
	v_fmac_f32_e64 v0, s37, s48                                // 000000001A0C: D52B0000 00006025
	v_fmac_f32_e64 v0, s40, s49                                // 000000001A14: D52B0000 00006228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A1C: BF8700A1
	v_fmac_f32_e64 v0, s38, s50                                // 000000001A20: D52B0000 00006426
	s_load_b32 s38, s[2:3], 0xfc8                              // 000000001A28: F4000981 F8000FC8
	v_fmac_f32_e64 v0, s39, s51                                // 000000001A30: D52B0000 00006627
	s_waitcnt lgkmcnt(0)                                       // 000000001A38: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A3C: BF8700A1
	v_fmac_f32_e64 v0, s36, s16                                // 000000001A40: D52B0000 00002024
	s_load_b64 s[36:37], s[2:3], 0x1440                        // 000000001A48: F4040901 F8001440
	v_fmac_f32_e64 v0, s12, s17                                // 000000001A50: D52B0000 0000220C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A58: BF870001
	v_fmac_f32_e64 v0, s13, s18                                // 000000001A5C: D52B0000 0000240D
	s_clause 0x2                                               // 000000001A64: BF850002
	s_load_b64 s[12:13], s[2:3], 0x1950                        // 000000001A68: F4040301 F8001950
	s_load_b64 s[16:17], s[2:3], 0x1e60                        // 000000001A70: F4040401 F8001E60
	s_load_b32 s72, s[2:3], 0x2370                             // 000000001A78: F4001201 F8002370
	v_fmac_f32_e64 v0, s38, s19                                // 000000001A80: D52B0000 00002626
	s_load_b64 s[18:19], s[2:3], s33 offset:0x90               // 000000001A88: F4040481 42000090
	s_waitcnt lgkmcnt(0)                                       // 000000001A90: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001A94: BF8700B1
	v_fmac_f32_e64 v0, s36, s20                                // 000000001A98: D52B0000 00002824
	s_load_b32 s20, s[2:3], s33 offset:0x50                    // 000000001AA0: F4000501 42000050
	s_movk_i32 s33, 0x37b0                                     // 000000001AA8: B02137B0
	v_fmac_f32_e64 v0, s37, s21                                // 000000001AAC: D52B0000 00002A25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AB4: BF870091
	v_fmac_f32_e64 v0, s35, s22                                // 000000001AB8: D52B0000 00002C23
	v_fmac_f32_e64 v0, s52, s23                                // 000000001AC0: D52B0000 00002E34
	s_clause 0x6                                               // 000000001AC8: BF850006
	s_load_b32 s22, s[2:3], s56 offset:0x50                    // 000000001ACC: F4000581 70000050
	s_load_b32 s23, s[2:3], s57 offset:0x50                    // 000000001AD4: F40005C1 72000050
	s_load_b32 s73, s[2:3], s34 offset:0x50                    // 000000001ADC: F4001241 44000050
	s_load_b32 s74, s[2:3], s60 offset:0x50                    // 000000001AE4: F4001281 78000050
	s_load_b32 s75, s[2:3], s62 offset:0x50                    // 000000001AEC: F40012C1 7C000050
	s_load_b32 s76, s[2:3], s63 offset:0x50                    // 000000001AF4: F4001301 7E000050
	s_load_b32 s77, s[2:3], s33 offset:0x50                    // 000000001AFC: F4001341 42000050
	v_fmac_f32_e64 v0, s53, s24                                // 000000001B04: D52B0000 00003035
	s_waitcnt lgkmcnt(0)                                       // 000000001B0C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B10: BF870001
	v_fmac_f32_e64 v0, s20, s25                                // 000000001B14: D52B0000 00003214
	s_clause 0x3                                               // 000000001B1C: BF850003
	s_load_b64 s[20:21], s[2:3], s56 offset:0x90               // 000000001B20: F4040501 70000090
	s_load_b64 s[52:53], s[2:3], s57 offset:0x90               // 000000001B28: F4040D01 72000090
	s_load_b64 s[54:55], s[2:3], s34 offset:0x90               // 000000001B30: F4040D81 44000090
	s_load_b32 s24, s[2:3], s56 offset:0x8                     // 000000001B38: F4000601 70000008
	v_fmac_f32_e64 v0, s18, s26                                // 000000001B40: D52B0000 00003412
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001B48: BF8700B1
	v_fmac_f32_e64 v0, s19, s27                                // 000000001B4C: D52B0000 00003613
	s_load_b64 s[18:19], s[2:3], s56 offset:0x48               // 000000001B54: F4040481 70000048
	s_load_b512 s[36:51], s[6:7], 0xc0                         // 000000001B5C: F4100903 F80000C0
	v_fmac_f32_e64 v0, s64, s28                                // 000000001B64: D52B0000 00003840
	s_movk_i32 s64, 0x3cc0                                     // 000000001B6C: B0403CC0
	s_clause 0x6                                               // 000000001B70: BF850006
	s_load_b32 s25, s[2:3], s57 offset:0x8                     // 000000001B74: F4000641 72000008
	s_load_b32 s79, s[2:3], s60 offset:0x8                     // 000000001B7C: F40013C1 78000008
	s_load_b32 s80, s[2:3], s62 offset:0x8                     // 000000001B84: F4001401 7C000008
	s_load_b32 s81, s[2:3], s63 offset:0x8                     // 000000001B8C: F4001441 7E000008
	s_load_b32 s82, s[2:3], s33 offset:0x8                     // 000000001B94: F4001481 42000008
	s_load_b32 s83, s[2:3], s64 offset:0x8                     // 000000001B9C: F40014C1 80000008
	s_load_b32 s84, s[2:3], s78 offset:0x8                     // 000000001BA4: F4001501 9C000008
	v_fmac_f32_e64 v0, s12, s29                                // 000000001BAC: D52B0000 00003A0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001BB4: BF870001
	v_fmac_f32_e64 v0, s13, s30                                // 000000001BB8: D52B0000 00003C0D
	s_clause 0x2                                               // 000000001BC0: BF850002
	s_load_b64 s[12:13], s[2:3], s57 offset:0x48               // 000000001BC4: F4040301 72000048
	s_load_b64 s[34:35], s[2:3], s34 offset:0x48               // 000000001BCC: F4040881 44000048
	s_load_b64 s[56:57], s[2:3], s60 offset:0x48               // 000000001BD4: F4040E01 78000048
	s_waitcnt lgkmcnt(0)                                       // 000000001BDC: BF89FC07
	v_fmac_f32_e64 v0, s24, s31                                // 000000001BE0: D52B0000 00003E18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BE8: BF870091
	v_fmac_f32_e64 v0, s18, s36                                // 000000001BEC: D52B0000 00004812
	v_fmac_f32_e64 v0, s19, s37                                // 000000001BF4: D52B0000 00004A13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BFC: BF870091
	v_fmac_f32_e64 v0, s22, s38                                // 000000001C00: D52B0000 00004C16
	v_fmac_f32_e64 v0, s20, s39                                // 000000001C08: D52B0000 00004E14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C10: BF870091
	v_fmac_f32_e64 v0, s21, s40                                // 000000001C14: D52B0000 00005015
	v_fmac_f32_e64 v0, s65, s41                                // 000000001C1C: D52B0000 00005241
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C24: BF870091
	v_fmac_f32_e64 v0, s16, s42                                // 000000001C28: D52B0000 00005410
	v_fmac_f32_e64 v0, s17, s43                                // 000000001C30: D52B0000 00005611
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C38: BF870091
	v_fmac_f32_e64 v0, s25, s44                                // 000000001C3C: D52B0000 00005819
	v_fmac_f32_e64 v0, s12, s45                                // 000000001C44: D52B0000 00005A0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C4C: BF870091
	v_fmac_f32_e64 v0, s13, s46                                // 000000001C50: D52B0000 00005C0D
	v_fmac_f32_e64 v0, s23, s47                                // 000000001C58: D52B0000 00005E17
	s_load_b512 s[16:31], s[6:7], 0x100                        // 000000001C60: F4100403 F8000100
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C68: BF870091
	v_fmac_f32_e64 v0, s52, s48                                // 000000001C6C: D52B0000 00006034
	v_fmac_f32_e64 v0, s53, s49                                // 000000001C74: D52B0000 00006235
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C7C: BF870091
	v_fmac_f32_e64 v0, s66, s50                                // 000000001C80: D52B0000 00006442
	v_fmac_f32_e64 v0, s72, s51                                // 000000001C88: D52B0000 00006648
	s_waitcnt lgkmcnt(0)                                       // 000000001C90: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C94: BF870091
	v_fmac_f32_e64 v0, s10, s16                                // 000000001C98: D52B0000 0000200A
	v_fmac_f32_e64 v0, s11, s17                                // 000000001CA0: D52B0000 0000220B
	s_load_b64 s[10:11], s[2:3], 0x2880                        // 000000001CA8: F4040281 F8002880
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CB0: BF870091
	v_fmac_f32_e64 v0, s34, s18                                // 000000001CB4: D52B0000 00002422
	v_fmac_f32_e64 v0, s35, s19                                // 000000001CBC: D52B0000 00002623
	s_clause 0x2                                               // 000000001CC4: BF850002
	s_load_b64 s[12:13], s[2:3], 0x2d90                        // 000000001CC8: F4040301 F8002D90
	s_load_b64 s[16:17], s[2:3], 0x32a0                        // 000000001CD0: F4040401 F80032A0
	s_load_b64 s[34:35], s[2:3], 0x37b0                        // 000000001CD8: F4040881 F80037B0
	s_load_b512 s[36:51], s[6:7], 0x140                        // 000000001CE0: F4100903 F8000140
	v_fmac_f32_e64 v0, s73, s20                                // 000000001CE8: D52B0000 00002849
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CF0: BF870091
	v_fmac_f32_e64 v0, s54, s21                                // 000000001CF4: D52B0000 00002A36
	v_fmac_f32_e64 v0, s55, s22                                // 000000001CFC: D52B0000 00002C37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D04: BF8700A1
	v_fmac_f32_e64 v0, s67, s23                                // 000000001D08: D52B0000 00002E43
	s_waitcnt lgkmcnt(0)                                       // 000000001D10: BF89FC07
	v_fmac_f32_e64 v0, s10, s24                                // 000000001D14: D52B0000 0000300A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001D1C: BF870001
	v_fmac_f32_e64 v0, s11, s25                                // 000000001D20: D52B0000 0000320B
	s_clause 0x3                                               // 000000001D28: BF850003
	s_load_b64 s[10:11], s[2:3], s60 offset:0x90               // 000000001D2C: F4040281 78000090
	s_load_b64 s[18:19], s[2:3], s62 offset:0x90               // 000000001D34: F4040481 7C000090
	s_load_b64 s[52:53], s[2:3], s63 offset:0x90               // 000000001D3C: F4040D01 7E000090
	s_load_b64 s[54:55], s[2:3], s33 offset:0x90               // 000000001D44: F4040D81 42000090
	v_fmac_f32_e64 v0, s79, s26                                // 000000001D4C: D52B0000 0000344F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D54: BF870091
	v_fmac_f32_e64 v0, s56, s27                                // 000000001D58: D52B0000 00003638
	v_fmac_f32_e64 v0, s57, s28                                // 000000001D60: D52B0000 00003839
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D68: BF8700A1
	v_fmac_f32_e64 v0, s74, s29                                // 000000001D6C: D52B0000 00003A4A
	s_waitcnt lgkmcnt(0)                                       // 000000001D74: BF89FC07
	v_fmac_f32_e64 v0, s10, s30                                // 000000001D78: D52B0000 00003C0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D80: BF8700A1
	v_fmac_f32_e64 v0, s11, s31                                // 000000001D84: D52B0000 00003E0B
	s_load_b64 s[10:11], s[2:3], s62 offset:0x48               // 000000001D8C: F4040281 7C000048
	v_fmac_f32_e64 v0, s68, s36                                // 000000001D94: D52B0000 00004844
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D9C: BF870091
	v_fmac_f32_e64 v0, s12, s37                                // 000000001DA0: D52B0000 00004A0C
	v_fmac_f32_e64 v0, s13, s38                                // 000000001DA8: D52B0000 00004C0D
	s_clause 0x2                                               // 000000001DB0: BF850002
	s_load_b64 s[12:13], s[2:3], s63 offset:0x48               // 000000001DB4: F4040301 7E000048
	s_load_b64 s[36:37], s[2:3], s33 offset:0x48               // 000000001DBC: F4040901 42000048
	s_load_b64 s[56:57], s[2:3], s64 offset:0x48               // 000000001DC4: F4040E01 80000048
	v_fmac_f32_e64 v0, s80, s39                                // 000000001DCC: D52B0000 00004E50
	s_waitcnt lgkmcnt(0)                                       // 000000001DD4: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DD8: BF870091
	v_fmac_f32_e64 v0, s10, s40                                // 000000001DDC: D52B0000 0000500A
	v_fmac_f32_e64 v0, s11, s41                                // 000000001DE4: D52B0000 0000520B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DEC: BF870091
	v_fmac_f32_e64 v0, s75, s42                                // 000000001DF0: D52B0000 0000544B
	v_fmac_f32_e64 v0, s18, s43                                // 000000001DF8: D52B0000 00005612
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E00: BF870091
	v_fmac_f32_e64 v0, s19, s44                                // 000000001E04: D52B0000 00005813
	v_fmac_f32_e64 v0, s69, s45                                // 000000001E0C: D52B0000 00005A45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E14: BF870091
	v_fmac_f32_e64 v0, s16, s46                                // 000000001E18: D52B0000 00005C10
	v_fmac_f32_e64 v0, s17, s47                                // 000000001E20: D52B0000 00005E11
	s_load_b512 s[16:31], s[6:7], 0x180                        // 000000001E28: F4100403 F8000180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E30: BF870091
	v_fmac_f32_e64 v0, s81, s48                                // 000000001E34: D52B0000 00006051
	v_fmac_f32_e64 v0, s12, s49                                // 000000001E3C: D52B0000 0000620C
	s_clause 0x1                                               // 000000001E44: BF850001
	s_load_b32 s12, s[2:3], s33 offset:0x98                    // 000000001E48: F4000301 42000098
	s_load_b64 s[10:11], s[2:3], 0x3cc0                        // 000000001E50: F4040281 F8003CC0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E58: BF870091
	v_fmac_f32_e64 v0, s13, s50                                // 000000001E5C: D52B0000 0000640D
	v_fmac_f32_e64 v0, s76, s51                                // 000000001E64: D52B0000 0000664C
	s_waitcnt lgkmcnt(0)                                       // 000000001E6C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E70: BF8700A1
	v_fmac_f32_e64 v0, s52, s16                                // 000000001E74: D52B0000 00002034
	s_movk_i32 s16, 0x46e0                                     // 000000001E7C: B01046E0
	v_fmac_f32_e64 v0, s53, s17                                // 000000001E80: D52B0000 00002235
	s_clause 0x2                                               // 000000001E88: BF850002
	s_load_b32 s17, s[2:3], s64 offset:0x98                    // 000000001E8C: F4000441 80000098
	s_load_b32 s33, s[2:3], s78 offset:0x98                    // 000000001E94: F4000841 9C000098
	s_load_b32 s60, s[2:3], s16 offset:0x98                    // 000000001E9C: F4000F01 20000098
	v_fmac_f32_e64 v0, s70, s18                                // 000000001EA4: D52B0000 00002446
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001EAC: BF870091
	v_fmac_f32_e64 v0, s34, s19                                // 000000001EB0: D52B0000 00002622
	v_fmac_f32_e64 v0, s35, s20                                // 000000001EB8: D52B0000 00002823
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001EC0: BF870091
	v_fmac_f32_e64 v0, s82, s21                                // 000000001EC4: D52B0000 00002A52
	v_fmac_f32_e64 v0, s36, s22                                // 000000001ECC: D52B0000 00002C24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001ED4: BF870091
	v_fmac_f32_e64 v0, s37, s23                                // 000000001ED8: D52B0000 00002E25
	v_fmac_f32_e64 v0, s77, s24                                // 000000001EE0: D52B0000 0000304D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001EE8: BF870091
	v_fmac_f32_e64 v0, s54, s25                                // 000000001EEC: D52B0000 00003236
	v_fmac_f32_e64 v0, s55, s26                                // 000000001EF4: D52B0000 00003437
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001EFC: BF870001
	v_fmac_f32_e64 v0, s12, s27                                // 000000001F00: D52B0000 0000360C
	s_clause 0x2                                               // 000000001F08: BF850002
	s_load_b64 s[12:13], s[2:3], 0x41d0                        // 000000001F0C: F4040301 F80041D0
	s_load_b64 s[34:35], s[2:3], 0x46e0                        // 000000001F14: F4040881 F80046E0
	s_load_b64 s[52:53], s[2:3], 0x4bf0                        // 000000001F1C: F4040D01 F8004BF0
	s_load_b512 s[36:51], s[6:7], 0x1c0                        // 000000001F24: F4100903 F80001C0
	v_fmac_f32_e64 v0, s10, s28                                // 000000001F2C: D52B0000 0000380A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001F34: BF870001
	v_fmac_f32_e64 v0, s11, s29                                // 000000001F38: D52B0000 00003A0B
	s_clause 0x3                                               // 000000001F40: BF850003
	s_load_b32 s18, s[2:3], s64 offset:0x50                    // 000000001F44: F4000481 80000050
	s_load_b64 s[10:11], s[2:3], s64 offset:0x90               // 000000001F4C: F4040281 80000090
	s_load_b32 s19, s[2:3], s78 offset:0x50                    // 000000001F54: F40004C1 9C000050
	s_load_b32 s62, s[2:3], s16 offset:0x50                    // 000000001F5C: F4000F81 20000050
	v_fmac_f32_e64 v0, s83, s30                                // 000000001F64: D52B0000 00003C53
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001F6C: BF8700A1
	v_fmac_f32_e64 v0, s56, s31                                // 000000001F70: D52B0000 00003E38
	s_waitcnt lgkmcnt(0)                                       // 000000001F78: BF89FC07
	v_fmac_f32_e64 v0, s57, s36                                // 000000001F7C: D52B0000 00004839
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001F84: BF870001
	v_fmac_f32_e64 v0, s18, s37                                // 000000001F88: D52B0000 00004A12
	s_clause 0x2                                               // 000000001F90: BF850002
	s_load_b64 s[36:37], s[2:3], s78 offset:0x90               // 000000001F94: F4040901 9C000090
	s_load_b64 s[54:55], s[2:3], s16 offset:0x90               // 000000001F9C: F4040D81 20000090
	s_load_b64 s[56:57], s[2:3], s58 offset:0x90               // 000000001FA4: F4040E01 74000090
	v_fmac_f32_e64 v0, s10, s38                                // 000000001FAC: D52B0000 00004C0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001FB4: BF8700A1
	v_fmac_f32_e64 v0, s11, s39                                // 000000001FB8: D52B0000 00004E0B
	s_load_b64 s[10:11], s[2:3], s78 offset:0x48               // 000000001FC0: F4040281 9C000048
	v_fmac_f32_e64 v0, s17, s40                                // 000000001FC8: D52B0000 00005011
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001FD0: BF870091
	v_fmac_f32_e64 v0, s12, s41                                // 000000001FD4: D52B0000 0000520C
	v_fmac_f32_e64 v0, s13, s42                                // 000000001FDC: D52B0000 0000540D
	s_load_b64 s[12:13], s[2:3], s16 offset:0x48               // 000000001FE4: F4040301 20000048
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001FEC: BF8700A1
	v_fmac_f32_e64 v0, s84, s43                                // 000000001FF0: D52B0000 00005654
	s_waitcnt lgkmcnt(0)                                       // 000000001FF8: BF89FC07
	v_fmac_f32_e64 v0, s10, s44                                // 000000001FFC: D52B0000 0000580A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002004: BF870091
	v_fmac_f32_e64 v0, s11, s45                                // 000000002008: D52B0000 00005A0B
	v_fmac_f32_e64 v0, s19, s46                                // 000000002010: D52B0000 00005C13
	s_load_b32 s2, s[2:3], s16 offset:0x8                      // 000000002018: F4000081 20000008
	s_load_b512 s[16:31], s[6:7], 0x200                        // 000000002020: F4100403 F8000200
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002028: BF870091
	v_fmac_f32_e64 v0, s36, s47                                // 00000000202C: D52B0000 00005E24
	v_fmac_f32_e64 v0, s37, s48                                // 000000002034: D52B0000 00006025
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000203C: BF870091
	v_fmac_f32_e64 v0, s33, s49                                // 000000002040: D52B0000 00006221
	v_fmac_f32_e64 v0, s34, s50                                // 000000002048: D52B0000 00006422
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002050: BF8700A1
	v_fmac_f32_e64 v0, s35, s51                                // 000000002054: D52B0000 00006623
	s_waitcnt lgkmcnt(0)                                       // 00000000205C: BF89FC07
	v_fmac_f32_e64 v0, s2, s16                                 // 000000002060: D52B0000 00002002
	s_lshl_b32 s2, s15, 8                                      // 000000002068: 8402880F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000206C: BF870099
	s_ashr_i32 s3, s2, 31                                      // 000000002070: 86039F02
	v_fmac_f32_e64 v0, s12, s17                                // 000000002074: D52B0000 0000220C
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000207C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002080: BF8700A9
	s_add_u32 s4, s4, s2                                       // 000000002084: 80040204
	s_addc_u32 s5, s5, s3                                      // 000000002088: 82050305
	v_fmac_f32_e64 v0, s13, s18                                // 00000000208C: D52B0000 0000240D
	s_lshl_b32 s2, s14, 4                                      // 000000002094: 8402840E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002098: BF870099
	s_ashr_i32 s3, s2, 31                                      // 00000000209C: 86039F02
	v_fmac_f32_e64 v0, s62, s19                                // 0000000020A0: D52B0000 0000263E
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000020A8: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000020AC: BF8700A9
	s_add_u32 s2, s4, s2                                       // 0000000020B0: 80020204
	s_addc_u32 s3, s5, s3                                      // 0000000020B4: 82030305
	v_fmac_f32_e64 v0, s54, s20                                // 0000000020B8: D52B0000 00002836
	s_add_u32 s0, s2, s0                                       // 0000000020C0: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000020C4: 82010103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020C8: BF870091
	v_fmac_f32_e64 v0, s55, s21                                // 0000000020CC: D52B0000 00002A37
	v_fmac_f32_e64 v0, s60, s22                                // 0000000020D4: D52B0000 00002C3C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020DC: BF870091
	v_fmac_f32_e64 v0, s52, s23                                // 0000000020E0: D52B0000 00002E34
	v_fmac_f32_e64 v0, s53, s24                                // 0000000020E8: D52B0000 00003035
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020F0: BF870091
	v_fmac_f32_e64 v0, s59, s25                                // 0000000020F4: D52B0000 0000323B
	v_fmac_f32_e64 v0, s8, s26                                 // 0000000020FC: D52B0000 00003408
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002104: BF870091
	v_fmac_f32_e64 v0, s9, s27                                 // 000000002108: D52B0000 00003609
	v_fmac_f32_e64 v0, s61, s28                                // 000000002110: D52B0000 0000383D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002118: BF870091
	v_fmac_f32_e64 v0, s56, s29                                // 00000000211C: D52B0000 00003A38
	v_fmac_f32_e64 v0, s57, s30                                // 000000002124: D52B0000 00003C39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000212C: BF870091
	v_fmac_f32_e64 v0, s71, s31                                // 000000002130: D52B0000 00003E47
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000002138: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 000000002140: DC6A0000 00000001
	s_nop 0                                                    // 000000002148: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000214C: BFB60003
	s_endpgm                                                   // 000000002150: BFB00000
