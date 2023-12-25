
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_150_150>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001708: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000170C: 86059F0F
	s_mul_i32 s8, s14, 0x96                                    // 000000001710: 9608FF0E 00000096
	s_lshl_b64 s[6:7], s[4:5], 2                               // 000000001718: 84868204
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001720: 80050602
	s_addc_u32 s6, s3, s7                                      // 000000001724: 82060703
	s_ashr_i32 s9, s8, 31                                      // 000000001728: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000172C: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001730: 84828208
	s_add_u32 s2, s5, s2                                       // 000000001734: 80020205
	s_addc_u32 s3, s6, s3                                      // 000000001738: 82030306
	s_load_b32 s5, s[2:3], null                                // 00000000173C: F4000141 F8000000
	s_mul_i32 s2, s15, 0x96                                    // 000000001744: 9602FF0F 00000096
	v_mov_b32_e32 v0, 0                                        // 00000000174C: 7E000280
	s_ashr_i32 s3, s2, 31                                      // 000000001750: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001754: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001758: 84828202
	s_add_u32 s2, s0, s2                                       // 00000000175C: 80020200
	s_addc_u32 s3, s1, s3                                      // 000000001760: 82030301
	s_ashr_i32 s15, s14, 31                                    // 000000001764: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001768: BF870499
	s_lshl_b64 s[0:1], s[14:15], 2                             // 00000000176C: 8480820E
	s_add_u32 s0, s2, s0                                       // 000000001770: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001774: 82010103
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	v_mov_b32_e32 v1, s5                                       // 00000000177C: 7E020205
	global_store_b32 v0, v1, s[0:1]                            // 000000001780: DC6A0000 00000100
	s_nop 0                                                    // 000000001788: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000178C: BFB60003
	s_endpgm                                                   // 000000001790: BFB00000
