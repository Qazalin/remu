
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_8_6_7_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s15, 11                                      // 000000001714: 96088B0F
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000172C: 80080806
	s_addc_u32 s9, s7, s9                                      // 000000001730: 82090907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001738: 84868202
	s_mul_i32 s2, s14, 5                                       // 00000000173C: 9602850E
	s_add_u32 s12, s8, s6                                      // 000000001740: 800C0608
	s_addc_u32 s13, s9, s7                                     // 000000001744: 820D0709
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001750: 84828202
	s_add_u32 s16, s0, s2                                      // 000000001754: 80100200
	s_addc_u32 s17, s1, s3                                     // 000000001758: 82110301
	s_load_b128 s[0:3], s[16:17], null                         // 00000000175C: F4080008 F8000000
	s_clause 0x1                                               // 000000001764: BF850001
	s_load_b128 s[8:11], s[12:13], null                        // 000000001768: F4080206 F8000000
	s_load_b32 s12, s[12:13], 0x10                             // 000000001770: F4000306 F8000010
	s_load_b32 s13, s[16:17], 0x10                             // 000000001778: F4000348 F8000010
	s_waitcnt lgkmcnt(0)                                       // 000000001780: BF89FC07
	v_fma_f32 v0, s8, s0, 0                                    // 000000001784: D6130000 02000008
	s_mul_i32 s0, s15, 42                                      // 00000000178C: 9600AA0F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001790: BF8704A1
	v_fmac_f32_e64 v0, s9, s1                                  // 000000001794: D52B0000 00000209
	s_ashr_i32 s1, s0, 31                                      // 00000000179C: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017A0: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017A4: BF8700C1
	v_fmac_f32_e64 v0, s10, s2                                 // 0000000017A8: D52B0000 0000040A
	s_mul_i32 s2, s14, 7                                       // 0000000017B0: 9602870E
	s_add_u32 s4, s4, s0                                       // 0000000017B4: 80040004
	s_addc_u32 s5, s5, s1                                      // 0000000017B8: 82050105
	v_fmac_f32_e64 v0, s11, s3                                 // 0000000017BC: D52B0000 0000060B
	s_ashr_i32 s3, s2, 31                                      // 0000000017C4: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C8: BF870099
	s_lshl_b64 s[0:1], s[2:3], 2                               // 0000000017CC: 84808202
	v_fmac_f32_e64 v0, s12, s13                                // 0000000017D0: D52B0000 00001A0C
	v_mov_b32_e32 v1, 0                                        // 0000000017D8: 7E020280
	s_add_u32 s0, s4, s0                                       // 0000000017DC: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017E0: 82010105
	s_add_u32 s0, s0, s6                                       // 0000000017E4: 80000600
	v_max_f32_e32 v0, 0, v0                                    // 0000000017E8: 20000080
	s_addc_u32 s1, s1, s7                                      // 0000000017EC: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 0000000017F0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017F8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017FC: BFB60003
	s_endpgm                                                   // 000000001800: BFB00000
