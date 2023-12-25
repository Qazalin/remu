
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4_4n18>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 00000000170C: 860F9F0E
	v_mov_b32_e32 v0, 0                                        // 000000001710: 7E000280
	s_lshl_b64 s[0:1], s[14:15], 1                             // 000000001714: 8480810E
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s6, s6, s0                                       // 00000000171C: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001720: 82070107
	s_lshl_b32 s2, s2, 2                                       // 000000001724: 84028202
	global_load_u16 v1, v0, s[6:7]                             // 000000001728: DC4A0000 01060000
	s_ashr_i32 s3, s2, 31                                      // 000000001730: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[2:3], s[2:3], 1                               // 000000001738: 84828102
	s_add_u32 s2, s4, s2                                       // 00000000173C: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001740: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001744: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001748: 82010103
	s_waitcnt vmcnt(0)                                         // 00000000174C: BF8903F7
	v_add_nc_u32_e32 v1, 1, v1                                 // 000000001750: 4A020281
	global_store_b16 v0, v1, s[0:1]                            // 000000001754: DC660000 00000100
	s_nop 0                                                    // 00000000175C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001760: BFB60003
	s_endpgm                                                   // 000000001764: BFB00000
