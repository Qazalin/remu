
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_22_28_5>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 0xc24                                   // 000000001708: 9606FF0F 00000C24
	s_mul_i32 s8, s14, 0x8c                                    // 000000001710: 9608FF0E 0000008C
	s_ashr_i32 s7, s6, 31                                      // 000000001718: 86079F06
	s_mov_b32 s4, s13                                          // 00000000171C: BE84000D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001720: 84868206
	v_mov_b32_e32 v1, 0                                        // 000000001724: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s5, s2, s6                                       // 00000000172C: 80050602
	s_addc_u32 s6, s3, s7                                      // 000000001730: 82060703
	s_ashr_i32 s9, s8, 31                                      // 000000001734: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000173C: 84828208
	s_add_u32 s7, s5, s2                                       // 000000001740: 80070205
	s_addc_u32 s6, s6, s3                                      // 000000001744: 82060306
	s_ashr_i32 s5, s13, 31                                     // 000000001748: 86059F0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001750: 84828204
	s_add_u32 s4, s7, s2                                       // 000000001754: 80040207
	s_addc_u32 s5, s6, s3                                      // 000000001758: 82050306
	s_clause 0x4                                               // 00000000175C: BF850004
	s_load_b32 s6, s[4:5], null                                // 000000001760: F4000182 F8000000
	s_load_b32 s7, s[4:5], 0x70                                // 000000001768: F40001C2 F8000070
	s_load_b32 s8, s[4:5], 0xe0                                // 000000001770: F4000202 F80000E0
	s_load_b32 s9, s[4:5], 0x150                               // 000000001778: F4000242 F8000150
	s_load_b32 s10, s[4:5], 0x1c0                              // 000000001780: F4000282 F80001C0
	s_mul_i32 s4, s15, 0x268                                   // 000000001788: 9604FF0F 00000268
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001790: BF870499
	s_ashr_i32 s5, s4, 31                                      // 000000001794: 86059F04
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001798: 84848204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 00000000179C: BF8700D9
	s_add_u32 s4, s0, s4                                       // 0000000017A0: 80040400
	s_addc_u32 s5, s1, s5                                      // 0000000017A4: 82050501
	s_waitcnt lgkmcnt(0)                                       // 0000000017A8: BF89FC07
	v_add_f32_e64 v0, s6, 0                                    // 0000000017AC: D5030000 00010006
	s_mul_i32 s6, s14, 28                                      // 0000000017B4: 96069C0E
	v_add_f32_e32 v0, s7, v0                                   // 0000000017B8: 06000007
	s_ashr_i32 s7, s6, 31                                      // 0000000017BC: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C0: BF870099
	s_lshl_b64 s[0:1], s[6:7], 2                               // 0000000017C4: 84808206
	v_add_f32_e32 v0, s8, v0                                   // 0000000017C8: 06000008
	s_add_u32 s0, s4, s0                                       // 0000000017CC: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017D0: 82010105
	s_add_u32 s0, s0, s2                                       // 0000000017D4: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000017D8: 82010301
	v_add_f32_e32 v0, s9, v0                                   // 0000000017DC: 06000009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E0: BF870091
	v_add_f32_e32 v0, s10, v0                                  // 0000000017E4: 0600000A
	v_mul_f32_e32 v0, 0x3e4ccccd, v0                           // 0000000017E8: 100000FF 3E4CCCCD
	global_store_b32 v1, v0, s[0:1]                            // 0000000017F0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017F8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017FC: BFB60003
	s_endpgm                                                   // 000000001800: BFB00000
