
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_7_7_4_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_i32 s0, s14, 9                                       // 000000001714: 9600890E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s1, s0, 31                                      // 00000000171C: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001720: BF870009
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001724: 84808200
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s10, s6, s0                                      // 00000000172C: 800A0006
	s_addc_u32 s7, s7, s1                                      // 000000001730: 82070107
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_mul_i32 s6, s15, 36                                      // 000000001738: 9606A40F
	s_lshl_b64 s[0:1], s[2:3], 2                               // 00000000173C: 84808202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s2, s10, s0                                      // 000000001744: 8002000A
	s_addc_u32 s3, s7, s1                                      // 000000001748: 82030107
	s_ashr_i32 s7, s6, 31                                      // 00000000174C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001750: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s6, s8, s6                                       // 000000001758: 80060608
	s_addc_u32 s7, s9, s7                                      // 00000000175C: 82070709
	s_load_b512 s[16:31], s[6:7], null                         // 000000001760: F4100403 F8000000
	s_clause 0x3                                               // 000000001768: BF850003
	s_load_b64 s[8:9], s[2:3], null                            // 00000000176C: F4040201 F8000000
	s_load_b32 s33, s[2:3], 0x8                                // 000000001774: F4000841 F8000008
	s_load_b64 s[10:11], s[2:3], 0x24                          // 00000000177C: F4040281 F8000024
	s_load_b32 s34, s[2:3], 0x2c                               // 000000001784: F4000881 F800002C
	s_waitcnt lgkmcnt(0)                                       // 00000000178C: BF89FC07
	v_fma_f32 v0, s8, s16, 0                                   // 000000001790: D6130000 02002008
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001798: BF870001
	v_fmac_f32_e64 v0, s9, s17                                 // 00000000179C: D52B0000 00002209
	s_clause 0x3                                               // 0000000017A4: BF850003
	s_load_b64 s[8:9], s[2:3], 0x144                           // 0000000017A8: F4040201 F8000144
	s_load_b64 s[12:13], s[2:3], 0x48                          // 0000000017B0: F4040301 F8000048
	s_load_b32 s16, s[2:3], 0x50                               // 0000000017B8: F4000401 F8000050
	s_load_b32 s17, s[2:3], 0x14c                              // 0000000017C0: F4000441 F800014C
	v_fmac_f32_e64 v0, s33, s18                                // 0000000017C8: D52B0000 00002421
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D0: BF870091
	v_fmac_f32_e64 v0, s10, s19                                // 0000000017D4: D52B0000 0000260A
	v_fmac_f32_e64 v0, s11, s20                                // 0000000017DC: D52B0000 0000280B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E4: BF8700A1
	v_fmac_f32_e64 v0, s34, s21                                // 0000000017E8: D52B0000 00002A22
	s_waitcnt lgkmcnt(0)                                       // 0000000017F0: BF89FC07
	v_fmac_f32_e64 v0, s12, s22                                // 0000000017F4: D52B0000 00002C0C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017FC: BF870001
	v_fmac_f32_e64 v0, s13, s23                                // 000000001800: D52B0000 00002E0D
	s_clause 0x1                                               // 000000001808: BF850001
	s_load_b64 s[10:11], s[2:3], 0x18c                         // 00000000180C: F4040281 F800018C
	s_load_b64 s[12:13], s[2:3], 0x168                         // 000000001814: F4040301 F8000168
	s_load_b512 s[36:51], s[6:7], 0x40                         // 00000000181C: F4100903 F8000040
	v_fmac_f32_e64 v0, s16, s24                                // 000000001824: D52B0000 00003010
	s_load_b32 s16, s[2:3], 0x170                              // 00000000182C: F4000401 F8000170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001834: BF870091
	v_fmac_f32_e64 v0, s8, s25                                 // 000000001838: D52B0000 00003208
	v_fmac_f32_e64 v0, s9, s26                                 // 000000001840: D52B0000 00003409
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001848: BF8700B1
	v_fmac_f32_e64 v0, s17, s27                                // 00000000184C: D52B0000 00003611
	s_load_b32 s17, s[2:3], 0x194                              // 000000001854: F4000441 F8000194
	s_waitcnt lgkmcnt(0)                                       // 00000000185C: BF89FC07
	v_fmac_f32_e64 v0, s12, s28                                // 000000001860: D52B0000 0000380C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001868: BF8700C1
	v_fmac_f32_e64 v0, s13, s29                                // 00000000186C: D52B0000 00003A0D
	s_clause 0x1                                               // 000000001874: BF850001
	s_load_b64 s[8:9], s[2:3], 0x2ac                           // 000000001878: F4040201 F80002AC
	s_load_b64 s[12:13], s[2:3], 0x288                         // 000000001880: F4040301 F8000288
	v_fmac_f32_e64 v0, s16, s30                                // 000000001888: D52B0000 00003C10
	s_load_b32 s16, s[2:3], 0x290                              // 000000001890: F4000401 F8000290
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001898: BF870091
	v_fmac_f32_e64 v0, s10, s31                                // 00000000189C: D52B0000 00003E0A
	v_fmac_f32_e64 v0, s11, s36                                // 0000000018A4: D52B0000 0000480B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000018AC: BF8700B1
	v_fmac_f32_e64 v0, s17, s37                                // 0000000018B0: D52B0000 00004A11
	s_load_b32 s17, s[2:3], 0x2b4                              // 0000000018B8: F4000441 F80002B4
	s_waitcnt lgkmcnt(0)                                       // 0000000018C0: BF89FC07
	v_fmac_f32_e64 v0, s12, s38                                // 0000000018C4: D52B0000 00004C0C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018CC: BF870001
	v_fmac_f32_e64 v0, s13, s39                                // 0000000018D0: D52B0000 00004E0D
	s_clause 0x2                                               // 0000000018D8: BF850002
	s_load_b64 s[12:13], s[2:3], 0x3cc                         // 0000000018DC: F4040301 F80003CC
	s_load_b64 s[10:11], s[2:3], 0x2d0                         // 0000000018E4: F4040281 F80002D0
	s_load_b32 s20, s[2:3], 0x3d4                              // 0000000018EC: F4000501 F80003D4
	v_fmac_f32_e64 v0, s16, s40                                // 0000000018F4: D52B0000 00005010
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018FC: BF8700A1
	v_fmac_f32_e64 v0, s8, s41                                 // 000000001900: D52B0000 00005208
	s_load_b32 s8, s[2:3], 0x2d8                               // 000000001908: F4000201 F80002D8
	v_fmac_f32_e64 v0, s9, s42                                 // 000000001910: D52B0000 00005409
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001918: BF870001
	v_fmac_f32_e64 v0, s17, s43                                // 00000000191C: D52B0000 00005611
	s_clause 0x1                                               // 000000001924: BF850001
	s_load_b64 s[16:17], s[2:3], 0x414                         // 000000001928: F4040401 F8000414
	s_load_b64 s[18:19], s[2:3], 0x3f0                         // 000000001930: F4040481 F80003F0
	s_waitcnt lgkmcnt(0)                                       // 000000001938: BF89FC07
	v_fmac_f32_e64 v0, s10, s44                                // 00000000193C: D52B0000 0000580A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001944: BF870091
	v_fmac_f32_e64 v0, s11, s45                                // 000000001948: D52B0000 00005A0B
	v_fmac_f32_e64 v0, s8, s46                                 // 000000001950: D52B0000 00005C08
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001958: BF870001
	v_fmac_f32_e64 v0, s12, s47                                // 00000000195C: D52B0000 00005E0C
	s_load_b32 s12, s[2:3], 0x3f8                              // 000000001964: F4000301 F80003F8
	s_load_b128 s[8:11], s[6:7], 0x80                          // 00000000196C: F4080203 F8000080
	s_load_b32 s7, s[2:3], 0x41c                               // 000000001974: F40001C1 F800041C
	s_mul_i32 s2, s15, 49                                      // 00000000197C: 9602B10F
	s_mul_i32 s6, s14, 7                                       // 000000001980: 9606870E
	v_fmac_f32_e64 v0, s13, s48                                // 000000001984: D52B0000 0000600D
	s_ashr_i32 s3, s2, 31                                      // 00000000198C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001990: BF870099
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001994: 84828202
	v_fmac_f32_e64 v0, s20, s49                                // 000000001998: D52B0000 00006214
	s_add_u32 s4, s4, s2                                       // 0000000019A0: 80040204
	s_addc_u32 s5, s5, s3                                      // 0000000019A4: 82050305
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019A8: BF870091
	v_fmac_f32_e64 v0, s18, s50                                // 0000000019AC: D52B0000 00006412
	v_fmac_f32_e64 v0, s19, s51                                // 0000000019B4: D52B0000 00006613
	s_waitcnt lgkmcnt(0)                                       // 0000000019BC: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019C0: BF870091
	v_fmac_f32_e64 v0, s12, s8                                 // 0000000019C4: D52B0000 0000100C
	v_fmac_f32_e64 v0, s16, s9                                 // 0000000019CC: D52B0000 00001210
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019D4: BF870091
	v_fmac_f32_e64 v0, s17, s10                                // 0000000019D8: D52B0000 00001411
	v_fmac_f32_e64 v0, s7, s11                                 // 0000000019E0: D52B0000 00001607
	s_ashr_i32 s7, s6, 31                                      // 0000000019E8: 86079F06
	v_mov_b32_e32 v1, 0                                        // 0000000019EC: 7E020280
	s_lshl_b64 s[2:3], s[6:7], 2                               // 0000000019F0: 84828206
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000019F4: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000019F8: 20000080
	s_add_u32 s2, s4, s2                                       // 0000000019FC: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001A00: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001A04: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001A08: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001A0C: DC6A0000 00000001
	s_nop 0                                                    // 000000001A14: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A18: BFB60003
	s_endpgm                                                   // 000000001A1C: BFB00000
