
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_8_6_10_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s15, 11                                      // 000000001714: 96088B0F
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s6, s6, s8                                       // 00000000172C: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001730: 82070907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001738: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000173C: BF8704B9
	s_add_u32 s6, s6, s2                                       // 000000001740: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001744: 82070307
	s_lshl_b32 s8, s14, 1                                      // 000000001748: 8408810E
	s_ashr_i32 s9, s8, 31                                      // 00000000174C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001754: 84888208
	s_add_u32 s0, s0, s8                                       // 000000001758: 80000800
	s_addc_u32 s1, s1, s9                                      // 00000000175C: 82010901
	s_load_b64 s[6:7], s[6:7], null                            // 000000001760: F4040183 F8000000
	s_load_b64 s[0:1], s[0:1], null                            // 000000001768: F4040000 F8000000
	s_mul_i32 s8, s15, 60                                      // 000000001770: 9608BC0F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001774: BF870499
	s_ashr_i32 s9, s8, 31                                      // 000000001778: 86099F08
	s_lshl_b64 s[8:9], s[8:9], 2                               // 00000000177C: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000001780: BF8700D9
	s_add_u32 s4, s4, s8                                       // 000000001784: 80040804
	s_addc_u32 s5, s5, s9                                      // 000000001788: 82050905
	s_waitcnt lgkmcnt(0)                                       // 00000000178C: BF89FC07
	v_fma_f32 v0, s6, s0, 0                                    // 000000001790: D6130000 02000006
	s_mul_i32 s0, s14, 10                                      // 000000001798: 96008A0E
	v_fmac_f32_e64 v0, s7, s1                                  // 00000000179C: D52B0000 00000207
	s_ashr_i32 s1, s0, 31                                      // 0000000017A4: 86019F00
	v_mov_b32_e32 v1, 0                                        // 0000000017A8: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017AC: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017B0: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000017B4: 20000080
	s_add_u32 s0, s4, s0                                       // 0000000017B8: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017BC: 82010105
	s_add_u32 s0, s0, s2                                       // 0000000017C0: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000017C4: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C8: DC6A0000 00000001
	s_nop 0                                                    // 0000000017D0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D4: BFB60003
	s_endpgm                                                   // 0000000017D8: BFB00000
