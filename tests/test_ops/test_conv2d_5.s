
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_10_6_3_2_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s14, 7                                       // 000000001714: 9608870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000172C: 80080806
	s_addc_u32 s9, s7, s9                                      // 000000001730: 82090907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001738: 84868202
	s_mul_i32 s2, s15, 12                                      // 00000000173C: 96028C0F
	s_add_u32 s8, s8, s6                                       // 000000001740: 80080608
	s_addc_u32 s9, s9, s7                                      // 000000001744: 82090709
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001750: 84828202
	s_add_u32 s0, s0, s2                                       // 000000001754: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001758: 82010301
	s_load_b256 s[16:23], s[0:1], null                         // 00000000175C: F40C0400 F8000000
	s_clause 0x3                                               // 000000001764: BF850003
	s_load_b64 s[2:3], s[8:9], null                            // 000000001768: F4040084 F8000000
	s_load_b64 s[10:11], s[8:9], 0x1c                          // 000000001770: F4040284 F800001C
	s_load_b64 s[12:13], s[8:9], 0x134                         // 000000001778: F4040304 F8000134
	s_load_b64 s[24:25], s[8:9], 0x150                         // 000000001780: F4040604 F8000150
	s_waitcnt lgkmcnt(0)                                       // 000000001788: BF89FC07
	v_fma_f32 v0, s2, s16, 0                                   // 00000000178C: D6130000 02002002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001794: BF8700A1
	v_fmac_f32_e64 v0, s3, s17                                 // 000000001798: D52B0000 00002203
	s_load_b128 s[0:3], s[0:1], 0x20                           // 0000000017A0: F4080000 F8000020
	v_fmac_f32_e64 v0, s10, s18                                // 0000000017A8: D52B0000 0000240A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017B0: BF8700C1
	v_fmac_f32_e64 v0, s11, s19                                // 0000000017B4: D52B0000 0000260B
	s_clause 0x1                                               // 0000000017BC: BF850001
	s_load_b64 s[10:11], s[8:9], 0x268                         // 0000000017C0: F4040284 F8000268
	s_load_b64 s[8:9], s[8:9], 0x284                           // 0000000017C8: F4040204 F8000284
	v_fmac_f32_e64 v0, s12, s20                                // 0000000017D0: D52B0000 0000280C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870091
	v_fmac_f32_e64 v0, s13, s21                                // 0000000017DC: D52B0000 00002A0D
	v_fmac_f32_e64 v0, s24, s22                                // 0000000017E4: D52B0000 00002C18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017EC: BF8700A1
	v_fmac_f32_e64 v0, s25, s23                                // 0000000017F0: D52B0000 00002E19
	s_waitcnt lgkmcnt(0)                                       // 0000000017F8: BF89FC07
	v_fmac_f32_e64 v0, s10, s0                                 // 0000000017FC: D52B0000 0000000A
	s_mul_i32 s0, s15, 60                                      // 000000001804: 9600BC0F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001808: BF8704A1
	v_fmac_f32_e64 v0, s11, s1                                 // 00000000180C: D52B0000 0000020B
	s_ashr_i32 s1, s0, 31                                      // 000000001814: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001818: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000181C: BF8700C1
	v_fmac_f32_e64 v0, s8, s2                                  // 000000001820: D52B0000 00000408
	s_mul_i32 s2, s14, 6                                       // 000000001828: 9602860E
	s_add_u32 s4, s4, s0                                       // 00000000182C: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001830: 82050105
	v_fmac_f32_e64 v0, s9, s3                                  // 000000001834: D52B0000 00000609
	s_ashr_i32 s3, s2, 31                                      // 00000000183C: 86039F02
	v_mov_b32_e32 v1, 0                                        // 000000001840: 7E020280
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001844: 84808202
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001848: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 00000000184C: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001850: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001854: 82010105
	s_add_u32 s0, s0, s6                                       // 000000001858: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000185C: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 000000001860: DC6A0000 00000001
	s_nop 0                                                    // 000000001868: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000186C: BFB60003
	s_endpgm                                                   // 000000001870: BFB00000
