
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_70_3_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s15, 0xe7                                    // 000000001714: 9608FF0F 000000E7
	s_mov_b32 s2, s13                                          // 00000000171C: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 000000001720: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001724: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001728: 84888208
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s8, s6, s8                                       // 000000001730: 80080806
	s_addc_u32 s9, s7, s9                                      // 000000001734: 82090907
	s_ashr_i32 s3, s13, 31                                     // 000000001738: 86039F0D
	s_lshl_b64 s[6:7], s[2:3], 2                               // 00000000173C: 84868202
	s_mul_i32 s2, s14, 6                                       // 000000001740: 9602860E
	s_add_u32 s8, s8, s6                                       // 000000001744: 80080608
	s_addc_u32 s9, s9, s7                                      // 000000001748: 82090709
	s_ashr_i32 s3, s2, 31                                      // 00000000174C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001754: 84828202
	s_add_u32 s10, s0, s2                                      // 000000001758: 800A0200
	s_addc_u32 s11, s1, s3                                     // 00000000175C: 820B0301
	s_load_b128 s[0:3], s[10:11], null                         // 000000001760: F4080005 F8000000
	s_clause 0x5                                               // 000000001768: BF850005
	s_load_b32 s12, s[8:9], null                               // 00000000176C: F4000304 F8000000
	s_load_b32 s13, s[8:9], 0x1c                               // 000000001774: F4000344 F800001C
	s_load_b32 s16, s[8:9], 0x134                              // 00000000177C: F4000404 F8000134
	s_load_b32 s17, s[8:9], 0x150                              // 000000001784: F4000444 F8000150
	s_load_b32 s18, s[8:9], 0x268                              // 00000000178C: F4000484 F8000268
	s_load_b32 s19, s[8:9], 0x284                              // 000000001794: F40004C4 F8000284
	s_load_b64 s[8:9], s[10:11], 0x10                          // 00000000179C: F4040205 F8000010
	s_waitcnt lgkmcnt(0)                                       // 0000000017A4: BF89FC07
	v_fma_f32 v0, s12, s0, 0                                   // 0000000017A8: D6130000 0200000C
	s_mul_i32 s0, s15, 0x1a4                                   // 0000000017B0: 9600FF0F 000001A4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017B8: BF8704A1
	v_fmac_f32_e64 v0, s13, s1                                 // 0000000017BC: D52B0000 0000020D
	s_ashr_i32 s1, s0, 31                                      // 0000000017C4: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017C8: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017CC: BF8700C1
	v_fmac_f32_e64 v0, s16, s2                                 // 0000000017D0: D52B0000 00000410
	s_mul_i32 s2, s14, 0x46                                    // 0000000017D8: 9602FF0E 00000046
	s_add_u32 s4, s4, s0                                       // 0000000017E0: 80040004
	s_addc_u32 s5, s5, s1                                      // 0000000017E4: 82050105
	v_fmac_f32_e64 v0, s17, s3                                 // 0000000017E8: D52B0000 00000611
	s_ashr_i32 s3, s2, 31                                      // 0000000017F0: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F4: BF870099
	s_lshl_b64 s[0:1], s[2:3], 2                               // 0000000017F8: 84808202
	v_fmac_f32_e64 v0, s18, s8                                 // 0000000017FC: D52B0000 00001012
	s_add_u32 s0, s4, s0                                       // 000000001804: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001808: 82010105
	s_add_u32 s0, s0, s6                                       // 00000000180C: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001810: 82010701
	v_fmac_f32_e64 v0, s19, s9                                 // 000000001814: D52B0000 00001213
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000181C: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001820: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 000000001828: DC6A0000 00000001
	s_nop 0                                                    // 000000001830: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001834: BFB60003
	s_endpgm                                                   // 000000001838: BFB00000
