
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_3_3_3>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 9                                       // 000000001708: 9606890F
	s_mul_i32 s8, s13, 3                                       // 00000000170C: 9608830D
	s_ashr_i32 s7, s6, 31                                      // 000000001710: 86079F06
	s_mov_b32 s4, s13                                          // 000000001714: BE84000D
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001718: 84868206
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001720: 80050602
	s_addc_u32 s9, s3, s7                                      // 000000001724: 82090703
	s_ashr_i32 s15, s14, 31                                    // 000000001728: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000172C: BF870499
	s_lshl_b64 s[2:3], s[14:15], 2                             // 000000001730: 8482820E
	s_add_u32 s5, s5, s2                                       // 000000001734: 80050205
	s_addc_u32 s10, s9, s3                                     // 000000001738: 820A0309
	s_ashr_i32 s9, s8, 31                                      // 00000000173C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001740: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001744: 84828208
	s_add_u32 s2, s5, s2                                       // 000000001748: 80020205
	s_addc_u32 s3, s10, s3                                     // 00000000174C: 8203030A
	s_load_b32 s2, s[2:3], null                                // 000000001750: F4000081 F8000000
	s_add_u32 s3, s0, s6                                       // 000000001758: 80030600
	s_mul_i32 s0, s14, 3                                       // 00000000175C: 9600830E
	s_addc_u32 s5, s1, s7                                      // 000000001760: 82050701
	s_ashr_i32 s1, s0, 31                                      // 000000001764: 86019F00
	v_mov_b32_e32 v0, 0                                        // 000000001768: 7E000280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000176C: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001770: BF8704B9
	s_add_u32 s3, s3, s0                                       // 000000001774: 80030003
	s_addc_u32 s6, s5, s1                                      // 000000001778: 82060105
	s_ashr_i32 s5, s13, 31                                     // 00000000177C: 86059F0D
	s_lshl_b64 s[0:1], s[4:5], 2                               // 000000001780: 84808204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_add_u32 s0, s3, s0                                       // 000000001788: 80000003
	s_addc_u32 s1, s6, s1                                      // 00000000178C: 82010106
	s_waitcnt lgkmcnt(0)                                       // 000000001790: BF89FC07
	v_mov_b32_e32 v1, s2                                       // 000000001794: 7E020202
	global_store_b32 v0, v1, s[0:1]                            // 000000001798: DC6A0000 00000100
	s_nop 0                                                    // 0000000017A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017A4: BFB60003
	s_endpgm                                                   // 0000000017A8: BFB00000
