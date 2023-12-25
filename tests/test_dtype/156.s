
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4_4n15>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 00000000170C: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001710: BF870009
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001714: 8480820E
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s6, s6, s0                                       // 00000000171C: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001720: 82070107
	s_lshl_b32 s2, s2, 2                                       // 000000001724: 84028202
	s_load_b32 s6, s[6:7], null                                // 000000001728: F4000183 F8000000
	s_ashr_i32 s3, s2, 31                                      // 000000001730: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001734: BF870009
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001738: 84828202
	s_waitcnt lgkmcnt(0)                                       // 00000000173C: BF89FC07
	s_add_i32 s6, s6, 1                                        // 000000001740: 81068106
	s_add_u32 s2, s4, s2                                       // 000000001744: 80020204
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s6              // 000000001748: CA100080 00000006
	s_addc_u32 s3, s5, s3                                      // 000000001750: 82030305
	s_add_u32 s0, s2, s0                                       // 000000001754: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001758: 82010103
	global_store_b32 v0, v1, s[0:1]                            // 00000000175C: DC6A0000 00000100
	s_nop 0                                                    // 000000001764: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001768: BFB60003
	s_endpgm                                                   // 00000000176C: BFB00000
