
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_32_7_7_32_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b64 s[6:7], s[0:1], 0x10                            // 000000001704: F4040180 F8000010
	s_load_b128 s[0:3], s[0:1], null                           // 00000000170C: F4080000 F8000000
	s_mul_i32 s8, s15, 0x120                                   // 000000001714: 9608FF0F 00000120
	s_mul_i32 s10, s14, 9                                      // 00000000171C: 960A890E
	s_ashr_i32 s9, s8, 31                                      // 000000001720: 86099F08
	s_mov_b32 s4, s13                                          // 000000001724: BE84000D
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001728: 84888208
	s_ashr_i32 s5, s13, 31                                     // 00000000172C: 86059F0D
	s_ashr_i32 s11, s10, 31                                    // 000000001730: 860B9F0A
	v_mov_b32_e32 v0, 0                                        // 000000001734: 7E000280
	s_waitcnt lgkmcnt(0)                                       // 000000001738: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000173C: 80080806
	s_addc_u32 s9, s7, s9                                      // 000000001740: 82090907
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001744: 84848204
	s_lshl_b64 s[6:7], s[10:11], 2                             // 000000001748: 8486820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000174C: BF870009
	s_add_u32 s6, s4, s6                                       // 000000001750: 80060604
	s_addc_u32 s7, s5, s7                                      // 000000001754: 82070705
	s_add_u32 s2, s6, s2                                       // 000000001758: 80020206
	s_addc_u32 s3, s7, s3                                      // 00000000175C: 82030307
	s_add_u32 s2, s2, 0x288                                    // 000000001760: 8002FF02 00000288
	s_addc_u32 s3, s3, 0                                       // 000000001768: 82038003
	s_mov_b64 s[6:7], 0                                        // 00000000176C: BE860180
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001770: BF870009
	s_add_u32 s10, s8, s6                                      // 000000001774: 800A0608
	s_addc_u32 s11, s9, s7                                     // 000000001778: 820B0709
	s_clause 0x3                                               // 00000000177C: BF850003
	s_load_b32 s20, s[2:3], -0x25c                             // 000000001780: F4000501 F81FFDA4
	s_load_b64 s[12:13], s[2:3], -0x264                        // 000000001788: F4040301 F81FFD9C
	s_load_b32 s21, s[2:3], -0x280                             // 000000001790: F4000541 F81FFD80
	s_load_b64 s[16:17], s[2:3], -0x288                        // 000000001798: F4040401 F81FFD78
	s_load_b512 s[36:51], s[10:11], null                       // 0000000017A0: F4100905 F8000000
	s_add_u32 s6, s6, 0x90                                     // 0000000017A8: 8006FF06 00000090
	s_addc_u32 s7, s7, 0                                       // 0000000017B0: 82078007
	s_waitcnt lgkmcnt(0)                                       // 0000000017B4: BF89FC07
	v_fmac_f32_e64 v0, s16, s36                                // 0000000017B8: D52B0000 00004810
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017C0: BF870001
	v_fmac_f32_e64 v0, s17, s37                                // 0000000017C4: D52B0000 00004A11
	s_clause 0x3                                               // 0000000017CC: BF850003
	s_load_b32 s22, s[2:3], -0x13c                             // 0000000017D0: F4000581 F81FFEC4
	s_load_b64 s[16:17], s[2:3], -0x144                        // 0000000017D8: F4040401 F81FFEBC
	s_load_b32 s23, s[2:3], -0x238                             // 0000000017E0: F40005C1 F81FFDC8
	s_load_b64 s[18:19], s[2:3], -0x240                        // 0000000017E8: F4040481 F81FFDC0
	v_fmac_f32_e64 v0, s21, s38                                // 0000000017F0: D52B0000 00004C15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F8: BF870091
	v_fmac_f32_e64 v0, s12, s39                                // 0000000017FC: D52B0000 00004E0C
	v_fmac_f32_e64 v0, s13, s40                                // 000000001804: D52B0000 0000500D
	s_clause 0x3                                               // 00000000180C: BF850003
	s_load_b32 s33, s[2:3], -0xf4                              // 000000001810: F4000841 F81FFF0C
	s_load_b64 s[12:13], s[2:3], -0xfc                         // 000000001818: F4040301 F81FFF04
	s_load_b32 s36, s[2:3], -0x118                             // 000000001820: F4000901 F81FFEE8
	s_load_b64 s[34:35], s[2:3], -0x120                        // 000000001828: F4040881 F81FFEE0
	v_fmac_f32_e64 v0, s20, s41                                // 000000001830: D52B0000 00005214
	s_waitcnt lgkmcnt(0)                                       // 000000001838: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000183C: BF870091
	v_fmac_f32_e64 v0, s18, s42                                // 000000001840: D52B0000 00005412
	v_fmac_f32_e64 v0, s19, s43                                // 000000001848: D52B0000 00005613
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001850: BF870091
	v_fmac_f32_e64 v0, s23, s44                                // 000000001854: D52B0000 00005817
	v_fmac_f32_e64 v0, s16, s45                                // 00000000185C: D52B0000 00005A10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001864: BF870091
	v_fmac_f32_e64 v0, s17, s46                                // 000000001868: D52B0000 00005C11
	v_fmac_f32_e64 v0, s22, s47                                // 000000001870: D52B0000 00005E16
	s_load_b512 s[16:31], s[10:11], 0x40                       // 000000001878: F4100405 F8000040
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001880: BF870091
	v_fmac_f32_e64 v0, s34, s48                                // 000000001884: D52B0000 00006022
	v_fmac_f32_e64 v0, s35, s49                                // 00000000188C: D52B0000 00006223
	s_load_b64 s[34:35], s[2:3], null                          // 000000001894: F4040881 F8000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000189C: BF8700A1
	v_fmac_f32_e64 v0, s36, s50                                // 0000000018A0: D52B0000 00006424
	s_load_b32 s36, s[2:3], 0x8                                // 0000000018A8: F4000901 F8000008
	v_fmac_f32_e64 v0, s12, s51                                // 0000000018B0: D52B0000 0000660C
	s_waitcnt lgkmcnt(0)                                       // 0000000018B8: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018BC: BF8700A1
	v_fmac_f32_e64 v0, s13, s16                                // 0000000018C0: D52B0000 0000200D
	s_load_b64 s[12:13], s[2:3], 0x24                          // 0000000018C8: F4040301 F8000024
	v_fmac_f32_e64 v0, s33, s17                                // 0000000018D0: D52B0000 00002221
	s_load_b32 s33, s[2:3], 0x2c                               // 0000000018D8: F4000841 F800002C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018E0: BF870091
	v_fmac_f32_e64 v0, s34, s18                                // 0000000018E4: D52B0000 00002422
	v_fmac_f32_e64 v0, s35, s19                                // 0000000018EC: D52B0000 00002623
	s_clause 0x3                                               // 0000000018F4: BF850003
	s_load_b32 s37, s[2:3], 0x14c                              // 0000000018F8: F4000941 F800014C
	s_load_b64 s[34:35], s[2:3], 0x144                         // 000000001900: F4040881 F8000144
	s_load_b32 s18, s[2:3], 0x50                               // 000000001908: F4000481 F8000050
	s_load_b64 s[16:17], s[2:3], 0x48                          // 000000001910: F4040401 F8000048
	v_fmac_f32_e64 v0, s36, s20                                // 000000001918: D52B0000 00002824
	s_waitcnt lgkmcnt(0)                                       // 000000001920: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001924: BF870091
	v_fmac_f32_e64 v0, s12, s21                                // 000000001928: D52B0000 00002A0C
	v_fmac_f32_e64 v0, s13, s22                                // 000000001930: D52B0000 00002C0D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001938: BF870001
	v_fmac_f32_e64 v0, s33, s23                                // 00000000193C: D52B0000 00002E21
	s_clause 0x3                                               // 000000001944: BF850003
	s_load_b32 s22, s[2:3], 0x194                              // 000000001948: F4000581 F8000194
	s_load_b64 s[12:13], s[2:3], 0x18c                         // 000000001950: F4040301 F800018C
	s_load_b32 s23, s[2:3], 0x170                              // 000000001958: F40005C1 F8000170
	s_load_b64 s[20:21], s[2:3], 0x168                         // 000000001960: F4040501 F8000168
	s_add_u32 s2, s2, 0x510                                    // 000000001968: 8002FF02 00000510
	s_addc_u32 s3, s3, 0                                       // 000000001970: 82038003
	s_cmpk_lg_i32 s6, 0x480                                    // 000000001974: B2060480
	v_fmac_f32_e64 v0, s16, s24                                // 000000001978: D52B0000 00003010
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001980: BF870091
	v_fmac_f32_e64 v0, s17, s25                                // 000000001984: D52B0000 00003211
	v_fmac_f32_e64 v0, s18, s26                                // 00000000198C: D52B0000 00003412
	s_load_b128 s[16:19], s[10:11], 0x80                       // 000000001994: F4080405 F8000080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000199C: BF870091
	v_fmac_f32_e64 v0, s34, s27                                // 0000000019A0: D52B0000 00003622
	v_fmac_f32_e64 v0, s35, s28                                // 0000000019A8: D52B0000 00003823
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019B0: BF8700A1
	v_fmac_f32_e64 v0, s37, s29                                // 0000000019B4: D52B0000 00003A25
	s_waitcnt lgkmcnt(0)                                       // 0000000019BC: BF89FC07
	v_fmac_f32_e64 v0, s20, s30                                // 0000000019C0: D52B0000 00003C14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019C8: BF870091
	v_fmac_f32_e64 v0, s21, s31                                // 0000000019CC: D52B0000 00003E15
	v_fmac_f32_e64 v0, s23, s16                                // 0000000019D4: D52B0000 00002017
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019DC: BF870091
	v_fmac_f32_e64 v0, s12, s17                                // 0000000019E0: D52B0000 0000220C
	v_fmac_f32_e64 v0, s13, s18                                // 0000000019E8: D52B0000 0000240D
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019F0: BF870001
	v_fmac_f32_e64 v0, s22, s19                                // 0000000019F4: D52B0000 00002616
	s_cbranch_scc1 65372                                       // 0000000019FC: BFA2FF5C <r_32_7_7_32_3_3+0x70>
	s_mul_i32 s2, s15, 49                                      // 000000001A00: 9602B10F
	s_mul_i32 s6, s14, 7                                       // 000000001A04: 9606870E
	s_ashr_i32 s3, s2, 31                                      // 000000001A08: 86039F02
	v_dual_max_f32 v0, v0, v0 :: v_dual_mov_b32 v1, 0          // 000000001A0C: CA900100 00000080
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001A14: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001A18: BF8704D9
	s_add_u32 s2, s0, s2                                       // 000000001A1C: 80020200
	s_addc_u32 s3, s1, s3                                      // 000000001A20: 82030301
	s_ashr_i32 s7, s6, 31                                      // 000000001A24: 86079F06
	v_max_f32_e32 v0, 0, v0                                    // 000000001A28: 20000080
	s_lshl_b64 s[0:1], s[6:7], 2                               // 000000001A2C: 84808206
	s_add_u32 s0, s2, s0                                       // 000000001A30: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001A34: 82010103
	s_add_u32 s0, s0, s4                                       // 000000001A38: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001A3C: 82010501
	global_store_b32 v1, v0, s[0:1]                            // 000000001A40: DC6A0000 00000001
	s_nop 0                                                    // 000000001A48: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A4C: BFB60003
	s_endpgm                                                   // 000000001A50: BFB00000
