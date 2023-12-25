
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_63_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s15, 0xe7                                    // 000000001714: 9608FF0F 000000E7
	s_mov_b32 s2, s13                                          // 00000000171C: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 000000001720: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001724: BF870009
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001728: 84888208
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s8, s6, s8                                       // 000000001730: 80080806
	s_addc_u32 s7, s7, s9                                      // 000000001734: 82070907
	s_ashr_i32 s3, s13, 31                                     // 000000001738: 86039F0D
	s_mul_i32 s6, s14, 9                                       // 00000000173C: 9606890E
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001740: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001744: BF8704B9
	s_add_u32 s8, s8, s2                                       // 000000001748: 80080208
	s_addc_u32 s9, s7, s3                                      // 00000000174C: 82090307
	s_ashr_i32 s7, s6, 31                                      // 000000001750: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001754: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001758: BF870009
	s_add_u32 s0, s0, s6                                       // 00000000175C: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001760: 82010701
	s_load_b256 s[16:23], s[0:1], null                         // 000000001764: F40C0400 F8000000
	s_clause 0x7                                               // 00000000176C: BF850007
	s_load_b32 s6, s[8:9], null                                // 000000001770: F4000184 F8000000
	s_load_b32 s7, s[8:9], 0x1c                                // 000000001778: F40001C4 F800001C
	s_load_b32 s10, s[8:9], 0x38                               // 000000001780: F4000284 F8000038
	s_load_b32 s11, s[8:9], 0x134                              // 000000001788: F40002C4 F8000134
	s_load_b32 s12, s[8:9], 0x150                              // 000000001790: F4000304 F8000150
	s_load_b32 s13, s[8:9], 0x16c                              // 000000001798: F4000344 F800016C
	s_load_b32 s24, s[8:9], 0x268                              // 0000000017A0: F4000604 F8000268
	s_load_b32 s25, s[8:9], 0x284                              // 0000000017A8: F4000644 F8000284
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_fma_f32 v0, s6, s16, 0                                   // 0000000017B4: D6130000 02002006
	s_mul_i32 s6, s14, 63                                      // 0000000017BC: 9606BF0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000017C0: BF8704C1
	v_fmac_f32_e64 v0, s7, s17                                 // 0000000017C4: D52B0000 00002207
	s_load_b32 s7, s[8:9], 0x2a0                               // 0000000017CC: F40001C4 F80002A0
	s_load_b32 s8, s[0:1], 0x20                                // 0000000017D4: F4000200 F8000020
	s_mul_i32 s0, s15, 0x17a                                   // 0000000017DC: 9600FF0F 0000017A
	s_ashr_i32 s1, s0, 31                                      // 0000000017E4: 86019F00
	v_fmac_f32_e64 v0, s10, s18                                // 0000000017E8: D52B0000 0000240A
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017F0: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017F4: BF8700A9
	s_add_u32 s4, s4, s0                                       // 0000000017F8: 80040004
	s_addc_u32 s5, s5, s1                                      // 0000000017FC: 82050105
	v_fmac_f32_e64 v0, s11, s19                                // 000000001800: D52B0000 0000260B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001808: BF870091
	v_fmac_f32_e64 v0, s12, s20                                // 00000000180C: D52B0000 0000280C
	v_fmac_f32_e64 v0, s13, s21                                // 000000001814: D52B0000 00002A0D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000181C: BF870091
	v_fmac_f32_e64 v0, s24, s22                                // 000000001820: D52B0000 00002C18
	v_fmac_f32_e64 v0, s25, s23                                // 000000001828: D52B0000 00002E19
	s_waitcnt lgkmcnt(0)                                       // 000000001830: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001834: BF870141
	v_fmac_f32_e64 v0, s7, s8                                  // 000000001838: D52B0000 00001007
	s_ashr_i32 s7, s6, 31                                      // 000000001840: 86079F06
	v_mov_b32_e32 v1, 0                                        // 000000001844: 7E020280
	s_lshl_b64 s[0:1], s[6:7], 2                               // 000000001848: 84808206
	v_max_f32_e32 v0, 0, v0                                    // 00000000184C: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001850: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001854: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001858: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000185C: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001860: DC6A0000 00000001
	s_nop 0                                                    // 000000001868: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000186C: BFB60003
	s_endpgm                                                   // 000000001870: BFB00000
