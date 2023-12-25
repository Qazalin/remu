
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4_4n6>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 00000000170C: 860F9F0E
	v_mov_b32_e32 v1, 0                                        // 000000001710: 7E020280
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001714: 8480820E
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s6, s6, s0                                       // 00000000171C: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001720: 82070107
	s_lshl_b32 s2, s2, 2                                       // 000000001724: 84028202
	s_load_b32 s6, s[6:7], null                                // 000000001728: F4000183 F8000000
	s_ashr_i32 s3, s2, 31                                      // 000000001730: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001738: 84828202
	s_add_u32 s2, s4, s2                                       // 00000000173C: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001740: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001744: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001748: 82010103
	s_waitcnt lgkmcnt(0)                                       // 00000000174C: BF89FC07
	v_add_f32_e64 v0, s6, 1.0                                  // 000000001750: D5030000 0001E406
	global_store_b32 v1, v0, s[0:1]                            // 000000001758: DC6A0000 00000001
	s_nop 0                                                    // 000000001760: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001764: BFB60003
	s_endpgm                                                   // 000000001768: BFB00000
