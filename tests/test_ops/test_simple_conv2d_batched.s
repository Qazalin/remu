
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_4_7_7_4_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_hi_i32 s2, s13, 0x92492493                           // 000000001714: 9702FF0D 92492493
	s_mul_i32 s0, s15, 0x144                                   // 00000000171C: 9600FF0F 00000144
	s_add_i32 s2, s2, s13                                      // 000000001724: 81020D02
	s_ashr_i32 s1, s0, 31                                      // 000000001728: 86019F00
	s_lshr_b32 s3, s2, 31                                      // 00000000172C: 85039F02
	s_ashr_i32 s2, s2, 2                                       // 000000001730: 86028202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001734: BF870009
	s_add_i32 s10, s2, s3                                      // 000000001738: 810A0302
	s_lshl_b64 s[2:3], s[0:1], 2                               // 00000000173C: 84828200
	s_mul_i32 s0, s10, 7                                       // 000000001740: 9600870A
	s_mul_i32 s10, s10, 9                                      // 000000001744: 960A890A
	s_sub_i32 s12, s13, s0                                     // 000000001748: 818C000D
	s_waitcnt lgkmcnt(0)                                       // 00000000174C: BF89FC07
	s_add_u32 s1, s6, s2                                       // 000000001750: 80010206
	s_addc_u32 s6, s7, s3                                      // 000000001754: 82060307
	s_ashr_i32 s11, s10, 31                                    // 000000001758: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000175C: BF8704D9
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001760: 8482820A
	s_mul_i32 s10, s14, 36                                     // 000000001764: 960AA40E
	s_add_u32 s1, s1, s2                                       // 000000001768: 80010201
	s_addc_u32 s7, s6, s3                                      // 00000000176C: 82070306
	s_ashr_i32 s13, s12, 31                                    // 000000001770: 860D9F0C
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000001774: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001778: BF8704B9
	s_add_u32 s6, s1, s2                                       // 00000000177C: 80060201
	s_addc_u32 s7, s7, s3                                      // 000000001780: 82070307
	s_ashr_i32 s11, s10, 31                                    // 000000001784: 860B9F0A
	s_lshl_b64 s[10:11], s[10:11], 2                           // 000000001788: 848A820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000178C: BF870009
	s_add_u32 s8, s8, s10                                      // 000000001790: 80080A08
	s_addc_u32 s9, s9, s11                                     // 000000001794: 82090B09
	s_load_b512 s[16:31], s[8:9], null                         // 000000001798: F4100404 F8000000
	s_clause 0x3                                               // 0000000017A0: BF850003
	s_load_b64 s[10:11], s[6:7], null                          // 0000000017A4: F4040283 F8000000
	s_load_b32 s1, s[6:7], 0x8                                 // 0000000017AC: F4000043 F8000008
	s_load_b64 s[12:13], s[6:7], 0x24                          // 0000000017B4: F4040303 F8000024
	s_load_b32 s33, s[6:7], 0x2c                               // 0000000017BC: F4000843 F800002C
	s_waitcnt lgkmcnt(0)                                       // 0000000017C4: BF89FC07
	v_fma_f32 v0, s10, s16, 0                                  // 0000000017C8: D6130000 0200200A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017D0: BF8700C1
	v_fmac_f32_e64 v0, s11, s17                                // 0000000017D4: D52B0000 0000220B
	s_clause 0x1                                               // 0000000017DC: BF850001
	s_load_b64 s[10:11], s[6:7], 0x144                         // 0000000017E0: F4040283 F8000144
	s_load_b64 s[16:17], s[6:7], 0x48                          // 0000000017E8: F4040403 F8000048
	v_fmac_f32_e64 v0, s1, s18                                 // 0000000017F0: D52B0000 00002401
	s_clause 0x1                                               // 0000000017F8: BF850001
	s_load_b32 s1, s[6:7], 0x50                                // 0000000017FC: F4000043 F8000050
	s_load_b32 s18, s[6:7], 0x14c                              // 000000001804: F4000483 F800014C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000180C: BF870091
	v_fmac_f32_e64 v0, s12, s19                                // 000000001810: D52B0000 0000260C
	v_fmac_f32_e64 v0, s13, s20                                // 000000001818: D52B0000 0000280D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001820: BF8700A1
	v_fmac_f32_e64 v0, s33, s21                                // 000000001824: D52B0000 00002A21
	s_waitcnt lgkmcnt(0)                                       // 00000000182C: BF89FC07
	v_fmac_f32_e64 v0, s16, s22                                // 000000001830: D52B0000 00002C10
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001838: BF870001
	v_fmac_f32_e64 v0, s17, s23                                // 00000000183C: D52B0000 00002E11
	s_clause 0x1                                               // 000000001844: BF850001
	s_load_b64 s[12:13], s[6:7], 0x18c                         // 000000001848: F4040303 F800018C
	s_load_b64 s[16:17], s[6:7], 0x168                         // 000000001850: F4040403 F8000168
	s_load_b512 s[36:51], s[8:9], 0x40                         // 000000001858: F4100904 F8000040
	v_fmac_f32_e64 v0, s1, s24                                 // 000000001860: D52B0000 00003001
	s_load_b32 s1, s[6:7], 0x170                               // 000000001868: F4000043 F8000170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001870: BF870091
	v_fmac_f32_e64 v0, s10, s25                                // 000000001874: D52B0000 0000320A
	v_fmac_f32_e64 v0, s11, s26                                // 00000000187C: D52B0000 0000340B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001884: BF8700B1
	v_fmac_f32_e64 v0, s18, s27                                // 000000001888: D52B0000 00003612
	s_load_b32 s18, s[6:7], 0x194                              // 000000001890: F4000483 F8000194
	s_waitcnt lgkmcnt(0)                                       // 000000001898: BF89FC07
	v_fmac_f32_e64 v0, s16, s28                                // 00000000189C: D52B0000 00003810
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000018A4: BF8700C1
	v_fmac_f32_e64 v0, s17, s29                                // 0000000018A8: D52B0000 00003A11
	s_clause 0x1                                               // 0000000018B0: BF850001
	s_load_b64 s[10:11], s[6:7], 0x2ac                         // 0000000018B4: F4040283 F80002AC
	s_load_b64 s[16:17], s[6:7], 0x288                         // 0000000018BC: F4040403 F8000288
	v_fmac_f32_e64 v0, s1, s30                                 // 0000000018C4: D52B0000 00003C01
	s_load_b32 s1, s[6:7], 0x290                               // 0000000018CC: F4000043 F8000290
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D4: BF870091
	v_fmac_f32_e64 v0, s12, s31                                // 0000000018D8: D52B0000 00003E0C
	v_fmac_f32_e64 v0, s13, s36                                // 0000000018E0: D52B0000 0000480D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000018E8: BF8700B1
	v_fmac_f32_e64 v0, s18, s37                                // 0000000018EC: D52B0000 00004A12
	s_load_b32 s18, s[6:7], 0x2b4                              // 0000000018F4: F4000483 F80002B4
	s_waitcnt lgkmcnt(0)                                       // 0000000018FC: BF89FC07
	v_fmac_f32_e64 v0, s16, s38                                // 000000001900: D52B0000 00004C10
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001908: BF870001
	v_fmac_f32_e64 v0, s17, s39                                // 00000000190C: D52B0000 00004E11
	s_clause 0x2                                               // 000000001914: BF850002
	s_load_b64 s[12:13], s[6:7], 0x3cc                         // 000000001918: F4040303 F80003CC
	s_load_b64 s[16:17], s[6:7], 0x2d0                         // 000000001920: F4040403 F80002D0
	s_load_b32 s20, s[6:7], 0x3d4                              // 000000001928: F4000503 F80003D4
	v_fmac_f32_e64 v0, s1, s40                                 // 000000001930: D52B0000 00005001
	s_load_b32 s1, s[6:7], 0x2d8                               // 000000001938: F4000043 F80002D8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001940: BF870091
	v_fmac_f32_e64 v0, s10, s41                                // 000000001944: D52B0000 0000520A
	v_fmac_f32_e64 v0, s11, s42                                // 00000000194C: D52B0000 0000540B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001954: BF8700A1
	v_fmac_f32_e64 v0, s18, s43                                // 000000001958: D52B0000 00005612
	s_waitcnt lgkmcnt(0)                                       // 000000001960: BF89FC07
	v_fmac_f32_e64 v0, s16, s44                                // 000000001964: D52B0000 00005810
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000196C: BF8700C1
	v_fmac_f32_e64 v0, s17, s45                                // 000000001970: D52B0000 00005A11
	s_clause 0x1                                               // 000000001978: BF850001
	s_load_b64 s[16:17], s[6:7], 0x414                         // 00000000197C: F4040403 F8000414
	s_load_b64 s[18:19], s[6:7], 0x3f0                         // 000000001984: F4040483 F80003F0
	v_fmac_f32_e64 v0, s1, s46                                 // 00000000198C: D52B0000 00005C01
	s_load_b32 s1, s[6:7], 0x3f8                               // 000000001994: F4000043 F80003F8
	s_load_b128 s[8:11], s[8:9], 0x80                          // 00000000199C: F4080204 F8000080
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000019A4: BF8704B1
	v_fmac_f32_e64 v0, s12, s47                                // 0000000019A8: D52B0000 00005E0C
	s_load_b32 s12, s[6:7], 0x41c                              // 0000000019B0: F4000303 F800041C
	s_mul_i32 s6, s15, 0xc4                                    // 0000000019B8: 9606FF0F 000000C4
	s_ashr_i32 s7, s6, 31                                      // 0000000019C0: 86079F06
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019C4: BF8700A1
	v_fmac_f32_e64 v0, s13, s48                                // 0000000019C8: D52B0000 0000600D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 0000000019D0: 84868206
	v_fmac_f32_e64 v0, s20, s49                                // 0000000019D4: D52B0000 00006214
	s_waitcnt lgkmcnt(0)                                       // 0000000019DC: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019E0: BF870091
	v_fmac_f32_e64 v0, s18, s50                                // 0000000019E4: D52B0000 00006412
	v_fmac_f32_e64 v0, s19, s51                                // 0000000019EC: D52B0000 00006613
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000019F4: BF8700C1
	v_fmac_f32_e64 v0, s1, s8                                  // 0000000019F8: D52B0000 00001001
	s_mul_i32 s8, s14, 49                                      // 000000001A00: 9608B10E
	s_add_u32 s1, s4, s6                                       // 000000001A04: 80010604
	s_addc_u32 s6, s5, s7                                      // 000000001A08: 82060705
	v_fmac_f32_e64 v0, s16, s9                                 // 000000001A0C: D52B0000 00001210
	s_ashr_i32 s9, s8, 31                                      // 000000001A14: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A18: BF870099
	s_lshl_b64 s[4:5], s[8:9], 2                               // 000000001A1C: 84848208
	v_fmac_f32_e64 v0, s17, s10                                // 000000001A20: D52B0000 00001411
	s_add_u32 s4, s1, s4                                       // 000000001A28: 80040401
	s_addc_u32 s5, s6, s5                                      // 000000001A2C: 82050506
	s_ashr_i32 s1, s0, 31                                      // 000000001A30: 86019F00
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A34: BF870001
	v_fmac_f32_e64 v0, s12, s11                                // 000000001A38: D52B0000 0000160C
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001A40: 84808200
	v_mov_b32_e32 v1, 0                                        // 000000001A44: 7E020280
	s_add_u32 s0, s4, s0                                       // 000000001A48: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001A4C: 82010105
	v_max_f32_e32 v0, 0, v0                                    // 000000001A50: 20000080
	s_add_u32 s0, s0, s2                                       // 000000001A54: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001A58: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001A5C: DC6A0000 00000001
	s_nop 0                                                    // 000000001A64: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A68: BFB60003
	s_endpgm                                                   // 000000001A6C: BFB00000
