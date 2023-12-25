
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4_2_3>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_lshl_b32 s6, s13, 2                                      // 000000001708: 8406820D
	s_mul_i32 s8, s14, 30                                      // 00000000170C: 96089E0E
	s_ashr_i32 s7, s6, 31                                      // 000000001710: 86079F06
	s_mov_b32 s4, s13                                          // 000000001714: BE84000D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001718: 84868206
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001720: 80050602
	s_addc_u32 s7, s3, s7                                      // 000000001724: 82070703
	s_ashr_i32 s9, s8, 31                                      // 000000001728: 86099F08
	s_mul_i32 s6, s15, 0x64                                    // 00000000172C: 9606FF0F 00000064
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001734: 84828208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001738: BF8704B9
	s_add_u32 s5, s5, s2                                       // 00000000173C: 80050205
	s_addc_u32 s8, s7, s3                                      // 000000001740: 82080307
	s_ashr_i32 s7, s6, 31                                      // 000000001744: 86079F06
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001748: 84828206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000174C: BF870009
	s_add_u32 s2, s5, s2                                       // 000000001750: 80020205
	s_addc_u32 s3, s8, s3                                      // 000000001754: 82030308
	s_load_b32 s6, s[2:3], null                                // 000000001758: F4000181 F8000000
	s_mul_i32 s2, s15, 6                                       // 000000001760: 9602860F
	v_mov_b32_e32 v0, 0                                        // 000000001764: 7E000280
	s_ashr_i32 s3, s2, 31                                      // 000000001768: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000176C: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001770: 84828202
	s_add_u32 s2, s0, s2                                       // 000000001774: 80020200
	s_mul_i32 s0, s14, 3                                       // 000000001778: 9600830E
	s_addc_u32 s3, s1, s3                                      // 00000000177C: 82030301
	s_ashr_i32 s1, s0, 31                                      // 000000001780: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001784: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001788: 84808200
	s_add_u32 s2, s2, s0                                       // 00000000178C: 80020002
	s_addc_u32 s3, s3, s1                                      // 000000001790: 82030103
	s_ashr_i32 s5, s13, 31                                     // 000000001794: 86059F0D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001798: BF870009
	s_lshl_b64 s[0:1], s[4:5], 2                               // 00000000179C: 84808204
	s_waitcnt lgkmcnt(0)                                       // 0000000017A0: BF89FC07
	v_mov_b32_e32 v1, s6                                       // 0000000017A4: 7E020206
	s_add_u32 s0, s2, s0                                       // 0000000017A8: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017AC: 82010103
	global_store_b32 v0, v1, s[0:1]                            // 0000000017B0: DC6A0000 00000100
	s_nop 0                                                    // 0000000017B8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017BC: BFB60003
	s_endpgm                                                   // 0000000017C0: BFB00000
