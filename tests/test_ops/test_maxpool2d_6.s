
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_1408_28_5>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mul_i32 s0, s15, 0x8c                                    // 000000001708: 9600FF0F 0000008C
	s_mov_b32 s2, s15                                          // 000000001710: BE82000F
	s_ashr_i32 s1, s0, 31                                      // 000000001714: 86019F00
	s_mul_i32 s2, s2, 28                                       // 000000001718: 96029C02
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000171C: 84808200
	v_mov_b32_e32 v1, 0                                        // 000000001720: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s3, s6, s0                                       // 000000001728: 80030006
	s_addc_u32 s7, s7, s1                                      // 00000000172C: 82070107
	s_ashr_i32 s15, s14, 31                                    // 000000001730: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001738: 8480820E
	s_add_u32 s6, s3, s0                                       // 00000000173C: 80060003
	s_addc_u32 s7, s7, s1                                      // 000000001740: 82070107
	s_clause 0x4                                               // 000000001744: BF850004
	s_load_b32 s3, s[6:7], null                                // 000000001748: F40000C3 F8000000
	s_load_b32 s8, s[6:7], 0x70                                // 000000001750: F4000203 F8000070
	s_load_b32 s9, s[6:7], 0xe0                                // 000000001758: F4000243 F80000E0
	s_load_b32 s10, s[6:7], 0x150                              // 000000001760: F4000283 F8000150
	s_load_b32 s6, s[6:7], 0x1c0                               // 000000001768: F4000183 F80001C0
	s_waitcnt lgkmcnt(0)                                       // 000000001770: BF89FC07
	v_max_f32_e64 v0, s3, s3                                   // 000000001774: D5100000 00000603
	s_ashr_i32 s3, s2, 31                                      // 00000000177C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001780: BF870099
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001784: 84828202
	v_max_f32_e32 v0, 0xff800000, v0                           // 000000001788: 200000FF FF800000
	s_add_u32 s2, s4, s2                                       // 000000001790: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001794: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001798: 80000002
	s_addc_u32 s1, s3, s1                                      // 00000000179C: 82010103
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017A0: D61C0000 04001009
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017A8: BF870001
	v_max3_f32 v0, s6, s10, v0                                 // 0000000017AC: D61C0000 04001406
	global_store_b32 v1, v0, s[0:1]                            // 0000000017B4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017BC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017C0: BFB60003
	s_endpgm                                                   // 0000000017C4: BFB00000
