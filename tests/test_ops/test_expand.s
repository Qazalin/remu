
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_12_2_6>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s6, s15, 6                                       // 000000001708: 9606860F
	s_mov_b32 s4, s13                                          // 00000000170C: BE84000D
	s_ashr_i32 s7, s6, 31                                      // 000000001710: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001714: BF8704D9
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001718: 84868206
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF89FC07
	s_add_u32 s6, s2, s6                                       // 000000001720: 80060602
	s_addc_u32 s7, s3, s7                                      // 000000001724: 82070703
	s_ashr_i32 s5, s13, 31                                     // 000000001728: 86059F0D
	s_lshl_b64 s[2:3], s[4:5], 2                               // 00000000172C: 84828204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001730: BF870009
	s_add_u32 s4, s6, s2                                       // 000000001734: 80040206
	s_addc_u32 s5, s7, s3                                      // 000000001738: 82050307
	s_load_b32 s6, s[4:5], null                                // 00000000173C: F4000182 F8000000
	s_mul_i32 s4, s15, 12                                      // 000000001744: 96048C0F
	v_mov_b32_e32 v0, 0                                        // 000000001748: 7E000280
	s_ashr_i32 s5, s4, 31                                      // 00000000174C: 86059F04
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001754: 84848204
	s_add_u32 s4, s0, s4                                       // 000000001758: 80040400
	s_mul_i32 s0, s14, 6                                       // 00000000175C: 9600860E
	s_addc_u32 s5, s1, s5                                      // 000000001760: 82050501
	s_ashr_i32 s1, s0, 31                                      // 000000001764: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001768: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000176C: 84808200
	s_add_u32 s0, s4, s0                                       // 000000001770: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001774: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001778: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000177C: 82010301
	s_waitcnt lgkmcnt(0)                                       // 000000001780: BF89FC07
	v_mov_b32_e32 v1, s6                                       // 000000001784: 7E020206
	global_store_b32 v0, v1, s[0:1]                            // 000000001788: DC6A0000 00000100
	s_nop 0                                                    // 000000001790: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001794: BFB60003
	s_endpgm                                                   // 000000001798: BFB00000
